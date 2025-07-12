"""Tests for doc2lora parsers."""

import json
import tempfile
import unittest
from pathlib import Path

from doc2lora.parsers import DocumentParser


class TestDocumentParser(unittest.TestCase):
    """Test cases for DocumentParser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = DocumentParser()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_parse_markdown(self):
        """Test parsing markdown files."""
        content = "# Test\n\nThis is a test markdown file."
        md_file = self.temp_path / "test.md"
        md_file.write_text(content)

        result = self.parser.parse_file(md_file)

        self.assertIsNotNone(result)
        self.assertEqual(result["content"], content)
        self.assertEqual(result["extension"], ".md")

    def test_parse_text(self):
        """Test parsing text files."""
        content = "This is a test text file."
        txt_file = self.temp_path / "test.txt"
        txt_file.write_text(content)

        result = self.parser.parse_file(txt_file)

        self.assertIsNotNone(result)
        self.assertEqual(result["content"], content)
        self.assertEqual(result["extension"], ".txt")

    def test_parse_json(self):
        """Test parsing JSON files."""
        data = {"test": "data", "number": 42}
        json_file = self.temp_path / "test.json"
        json_file.write_text(json.dumps(data))

        result = self.parser.parse_file(json_file)

        self.assertIsNotNone(result)
        self.assertIn('"test": "data"', result["content"])
        self.assertEqual(result["extension"], ".json")

    def test_parse_csv(self):
        """Test parsing CSV files."""
        content = "name,age\nJohn,30\nJane,25"
        csv_file = self.temp_path / "test.csv"
        csv_file.write_text(content)

        result = self.parser.parse_file(csv_file)

        self.assertIsNotNone(result)
        self.assertIn("name, age", result["content"])
        self.assertEqual(result["extension"], ".csv")

    def test_parse_directory(self):
        """Test parsing entire directories."""
        # Create test files
        (self.temp_path / "test1.md").write_text("# Test 1")
        (self.temp_path / "test2.txt").write_text("Test 2 content")
        (self.temp_path / "test3.json").write_text('{"test": 3}')

        # Create subdirectory with file
        subdir = self.temp_path / "subdir"
        subdir.mkdir()
        (subdir / "test4.md").write_text("# Test 4")

        results = self.parser.parse_directory(str(self.temp_path))

        self.assertEqual(len(results), 4)
        filenames = [r["filename"] for r in results]
        self.assertIn("test1.md", filenames)
        self.assertIn("test2.txt", filenames)
        self.assertIn("test3.json", filenames)
        self.assertIn("test4.md", filenames)

    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        unsupported_file = self.temp_path / "test.xyz"
        unsupported_file.write_text("This should not be parsed")

        result = self.parser.parse_file(unsupported_file)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
