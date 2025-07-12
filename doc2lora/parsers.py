"""Document parsers for various file formats."""

import csv
import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parser for various document formats."""

    SUPPORTED_EXTENSIONS = {
        ".md",
        ".txt",
        ".pdf",
        ".html",
        ".docx",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".tex",
    }

    def __init__(self):
        """Initialize the document parser."""
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if optional dependencies are available."""
        missing_deps = []

        if PyPDF2 is None:
            missing_deps.append("PyPDF2 (for PDF support)")
        if Document is None:
            missing_deps.append("python-docx (for DOCX support)")
        if BeautifulSoup is None:
            missing_deps.append("beautifulsoup4 (for HTML support)")
        if yaml is None:
            missing_deps.append("pyyaml (for YAML support)")

        if missing_deps:
            logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")

    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Recursively parse all supported documents in a directory.

        Args:
            directory_path: Path to the directory to scan

        Returns:
            List of parsed documents with metadata
        """
        documents = []
        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Recursively find all supported files
        for file_path in directory_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ):
                try:
                    doc = self.parse_file(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}")
                    continue

        return documents

    def parse_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a single file based on its extension.

        Args:
            file_path: Path to the file to parse

        Returns:
            Parsed document with metadata or None if parsing failed
        """
        extension = file_path.suffix.lower()

        try:
            if extension == ".md":
                content = self._parse_markdown(file_path)
            elif extension == ".txt" or extension == "":
                content = self._parse_text(file_path)
            elif extension == ".pdf":
                content = self._parse_pdf(file_path)
            elif extension == ".html":
                content = self._parse_html(file_path)
            elif extension == ".docx":
                content = self._parse_docx(file_path)
            elif extension == ".csv":
                content = self._parse_csv(file_path)
            elif extension == ".json":
                content = self._parse_json(file_path)
            elif extension in [".yaml", ".yml"]:
                content = self._parse_yaml(file_path)
            elif extension == ".xml":
                content = self._parse_xml(file_path)
            elif extension == ".tex":
                content = self._parse_latex(file_path)
            else:
                logger.warning(f"Unsupported file type: {extension}")
                return None

            if content:
                return {
                    "content": content,
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "extension": extension,
                    "size": file_path.stat().st_size,
                }

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def _parse_markdown(self, file_path: Path) -> str:
        """Parse Markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_text(self, file_path: Path) -> str:
        """Parse text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_pdf(self, file_path: Path) -> str:
        """Parse PDF file."""
        if PyPDF2 is None:
            logger.error("PyPDF2 not installed. Cannot parse PDF files.")
            return ""

        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _parse_html(self, file_path: Path) -> str:
        """Parse HTML file."""
        if BeautifulSoup is None:
            logger.error("BeautifulSoup not installed. Cannot parse HTML files.")
            return ""

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text()

    def _parse_docx(self, file_path: Path) -> str:
        """Parse Word document."""
        if Document is None:
            logger.error("python-docx not installed. Cannot parse DOCX files.")
            return ""

        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _parse_csv(self, file_path: Path) -> str:
        """Parse CSV file."""
        text = ""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                text += ", ".join(row) + "\n"
        return text

    def _parse_json(self, file_path: Path) -> str:
        """Parse JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return json.dumps(data, indent=2)

    def _parse_yaml(self, file_path: Path) -> str:
        """Parse YAML file."""
        if yaml is None:
            logger.error("PyYAML not installed. Cannot parse YAML files.")
            return ""

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return yaml.dump(data, default_flow_style=False)

    def _parse_xml(self, file_path: Path) -> str:
        """Parse XML file."""
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ET.parse(f)
            root = tree.getroot()
            return ET.tostring(root, encoding="unicode")

    def _parse_latex(self, file_path: Path) -> str:
        """Parse LaTeX file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
