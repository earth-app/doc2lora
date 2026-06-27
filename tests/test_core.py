"""Tests for doc2lora core functionality."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from doc2lora.core import _process_input_data, convert, convert_from_r2


class TestProcessInputData(unittest.TestCase):
    """Test cases for in-memory input handling."""

    def test_single_string(self):
        docs = _process_input_data("hello world")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "hello world")
        self.assertTrue(docs[0]["filepath"].startswith("memory://"))

    def test_list_of_strings(self):
        docs = _process_input_data(["a", "b", "c"])
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[1]["content"], "b")

    def test_single_bytes(self):
        docs = _process_input_data(b"byte content")
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "byte content")

    def test_list_of_bytes(self):
        docs = _process_input_data([b"one", b"two"])
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["content"], "one")

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            _process_input_data(12345)


class TestConvertFromR2(unittest.TestCase):
    """Test cases for the R2 conversion wrapper."""

    @patch("doc2lora.core.cleanup_temp_directory")
    @patch("doc2lora.core.convert")
    @patch("doc2lora.core.download_from_r2_bucket")
    def test_delegates_and_cleans_up(self, mock_download, mock_convert, mock_cleanup):
        mock_download.return_value = "/tmp/doc2lora_r2_xyz"
        mock_convert.return_value = "adapter.json"

        result = convert_from_r2(
            bucket_name="bkt",
            output_path="adapter.json",
            aws_access_key_id="k",
            aws_secret_access_key="s",
            endpoint_url="https://acct.r2.cloudflarestorage.com",
        )

        self.assertEqual(result, "adapter.json")
        mock_convert.assert_called_once()
        # delegated with the downloaded temp dir as documents_path
        self.assertEqual(
            mock_convert.call_args.kwargs["documents_path"], "/tmp/doc2lora_r2_xyz"
        )
        # temp dir cleaned up afterward
        mock_cleanup.assert_called_once_with("/tmp/doc2lora_r2_xyz")


class TestCore(unittest.TestCase):
    """Test cases for core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create sample documents
        (self.temp_path / "test1.md").write_text("# Test Document 1\n\nContent here.")
        (self.temp_path / "test2.txt").write_text("This is test document 2.")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("doc2lora.lora_trainer.LoRATrainer")
    def test_convert_basic(self, mock_trainer_class):
        """Test basic convert functionality."""
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer.save_adapter.return_value = "test_adapter.json"
        mock_trainer_class.return_value = mock_trainer

        output_path = str(self.temp_path / "output.json")

        result = convert(documents_path=str(self.temp_path), output_path=output_path)

        # Verify trainer was called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_trainer.save_adapter.assert_called_once_with(output_path)

        self.assertEqual(result, "test_adapter.json")

    def test_convert_empty_directory(self):
        """Test convert with empty directory."""
        empty_dir = self.temp_path / "empty"
        empty_dir.mkdir()

        with self.assertRaises(ValueError):
            convert(documents_path=str(empty_dir), output_path="output.json")

    def test_convert_nonexistent_directory(self):
        """Test convert with nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            convert(documents_path="nonexistent_directory", output_path="output.json")


if __name__ == "__main__":
    unittest.main()
