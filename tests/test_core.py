"""Tests for doc2lora core functionality."""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from doc2lora.core import convert


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
    
    @patch('doc2lora.core.LoRATrainer')
    def test_convert_basic(self, mock_trainer_class):
        """Test basic convert functionality."""
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer.save_adapter.return_value = "test_adapter.json"
        mock_trainer_class.return_value = mock_trainer
        
        output_path = str(self.temp_path / "output.json")
        
        result = convert(
            documents_path=str(self.temp_path),
            output_path=output_path
        )
        
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
            convert(
                documents_path=str(empty_dir),
                output_path="output.json"
            )
    
    def test_convert_nonexistent_directory(self):
        """Test convert with nonexistent directory."""
        with self.assertRaises(FileNotFoundError):
            convert(
                documents_path="nonexistent_directory",
                output_path="output.json"
            )


if __name__ == '__main__':
    unittest.main()
