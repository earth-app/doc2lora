"""Document parsers for various file formats."""

import csv
import json
import logging
import os
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile
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
        ".zip",
        ".tar",
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
        ".tgz",
        ".tbz2",
        ".txz",
    }

    # Extensions of files that can be contained within archives
    DOCUMENT_EXTENSIONS = {
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

        # Handle compound extensions for tar files
        if file_path.name.lower().endswith(('.tar.gz', '.tar.bz2', '.tar.xz')):
            if file_path.name.lower().endswith('.tar.gz') or file_path.name.lower().endswith('.tgz'):
                extension = '.tar.gz'
            elif file_path.name.lower().endswith('.tar.bz2') or file_path.name.lower().endswith('.tbz2'):
                extension = '.tar.bz2'
            elif file_path.name.lower().endswith('.tar.xz') or file_path.name.lower().endswith('.txz'):
                extension = '.tar.xz'

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
            elif extension == ".zip":
                content = self._parse_zip(file_path)
            elif extension in [".tar", ".tar.gz", ".tar.bz2", ".tar.xz"]:
                content = self._parse_tar(file_path)
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

    def _parse_zip(self, file_path: Path) -> str:
        """Parse ZIP archive by extracting and parsing supported documents."""
        content_parts = []

        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Get list of files in the archive
                file_list = zip_ref.namelist()
                content_parts.append(f"=== ZIP Archive: {file_path.name} ===")
                content_parts.append(f"Contains {len(file_list)} files:")

                # Create a temporary directory to extract files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    for file_info in zip_ref.infolist():
                        # Skip directories and hidden files
                        if file_info.is_dir() or file_info.filename.startswith('.'):
                            continue

                        file_name = Path(file_info.filename)
                        extension = file_name.suffix.lower()

                        # Only process files with supported document extensions
                        if extension in self.DOCUMENT_EXTENSIONS:
                            try:
                                # Extract the file to temp directory
                                extracted_path = zip_ref.extract(file_info, temp_dir)
                                extracted_file = Path(extracted_path)

                                # Parse the extracted file
                                parsed_doc = self._parse_extracted_file(extracted_file, file_info.filename)
                                if parsed_doc:
                                    content_parts.append(f"\n--- File: {file_info.filename} ---")
                                    content_parts.append(parsed_doc)

                            except Exception as e:
                                logger.warning(f"Could not parse {file_info.filename} from ZIP: {e}")
                                content_parts.append(f"\n--- File: {file_info.filename} (parsing failed) ---")
                        else:
                            content_parts.append(f"  • {file_info.filename} (unsupported format)")

        except Exception as e:
            logger.error(f"Error reading ZIP file {file_path}: {e}")
            return f"Error reading ZIP file: {e}"

        return "\n".join(content_parts)

    def _parse_tar(self, file_path: Path) -> str:
        """Parse TAR archive by extracting and parsing supported documents."""
        content_parts = []

        try:
            # Determine the compression mode
            if file_path.name.lower().endswith(('.tar.gz', '.tgz')):
                mode = 'r:gz'
            elif file_path.name.lower().endswith(('.tar.bz2', '.tbz2')):
                mode = 'r:bz2'
            elif file_path.name.lower().endswith(('.tar.xz', '.txz')):
                mode = 'r:xz'
            else:
                mode = 'r'

            with tarfile.open(file_path, mode) as tar_ref:
                # Get list of files in the archive
                members = tar_ref.getmembers()
                file_members = [m for m in members if m.isfile()]

                content_parts.append(f"=== TAR Archive: {file_path.name} ===")
                content_parts.append(f"Contains {len(file_members)} files:")

                # Create a temporary directory to extract files
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    for member in file_members:
                        # Skip hidden files
                        if Path(member.name).name.startswith('.'):
                            continue

                        file_name = Path(member.name)
                        extension = file_name.suffix.lower()

                        # Only process files with supported document extensions
                        if extension in self.DOCUMENT_EXTENSIONS:
                            try:
                                # Extract the file to temp directory
                                tar_ref.extract(member, temp_dir)
                                extracted_file = temp_path / member.name

                                # Parse the extracted file
                                parsed_doc = self._parse_extracted_file(extracted_file, member.name)
                                if parsed_doc:
                                    content_parts.append(f"\n--- File: {member.name} ---")
                                    content_parts.append(parsed_doc)

                            except Exception as e:
                                logger.warning(f"Could not parse {member.name} from TAR: {e}")
                                content_parts.append(f"\n--- File: {member.name} (parsing failed) ---")
                        else:
                            content_parts.append(f"  • {member.name} (unsupported format)")

        except Exception as e:
            logger.error(f"Error reading TAR file {file_path}: {e}")
            return f"Error reading TAR file: {e}"

        return "\n".join(content_parts)

    def _parse_extracted_file(self, file_path: Path, original_name: str) -> Optional[str]:
        """Parse an extracted file from an archive."""
        extension = file_path.suffix.lower()

        try:
            if extension == ".md":
                return self._parse_markdown(file_path)
            elif extension == ".txt":
                return self._parse_text(file_path)
            elif extension == ".pdf":
                return self._parse_pdf(file_path)
            elif extension == ".html":
                return self._parse_html(file_path)
            elif extension == ".docx":
                return self._parse_docx(file_path)
            elif extension == ".csv":
                return self._parse_csv(file_path)
            elif extension == ".json":
                return self._parse_json(file_path)
            elif extension in [".yaml", ".yml"]:
                return self._parse_yaml(file_path)
            elif extension == ".xml":
                return self._parse_xml(file_path)
            elif extension == ".tex":
                return self._parse_latex(file_path)
            else:
                return None

        except Exception as e:
            logger.warning(f"Error parsing extracted file {original_name}: {e}")
            return None
