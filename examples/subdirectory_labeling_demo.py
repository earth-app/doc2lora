#!/usr/bin/env python3
"""
Demonstration of subdirectory labeling feature in doc2lora.

This example shows how doc2lora uses subdirectory structure to automatically
label documents, which is useful for organizing training data by category.
"""

import json
import tempfile
from pathlib import Path

from doc2lora.parsers import DocumentParser


def create_demo_structure():
    """Create a demo directory structure with labeled subdirectories."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    base_path = Path(temp_dir)

    print(f"Creating demo structure in: {base_path}")

    # Create subdirectories representing different document categories
    categories = {
        "legal": [
            (
                "contract.txt",
                "This is a legal contract document with terms and conditions.",
            ),
            (
                "agreement.md",
                "# Legal Agreement\n\nThis document outlines the legal agreement between parties.",
            ),
        ],
        "technical": [
            (
                "api_spec.md",
                "# API Specification\n\n## Endpoints\n\n- GET /users\n- POST /users",
            ),
            ("readme.txt", "Technical documentation for the software project."),
        ],
        "marketing": [
            ("campaign.txt", "Marketing campaign strategy for Q4 2025."),
            (
                "brochure.md",
                "# Product Brochure\n\nOur amazing product features include...",
            ),
        ],
        "support": [
            (
                "faq.txt",
                "Frequently Asked Questions\n\nQ: How do I install?\nA: Run pip install...",
            ),
            (
                "troubleshooting.md",
                "# Troubleshooting Guide\n\n## Common Issues\n\n1. Installation problems",
            ),
        ],
    }

    # Create files in subdirectories
    for category, files in categories.items():
        category_dir = base_path / category
        category_dir.mkdir(exist_ok=True)

        for filename, content in files:
            file_path = category_dir / filename
            file_path.write_text(content, encoding="utf-8")
            print(f"  Created: {category}/{filename}")

    # Create a root-level file
    root_file = base_path / "overview.txt"
    root_file.write_text(
        "This is a root-level document providing an overview of all categories.",
        encoding="utf-8",
    )
    print(f"  Created: overview.txt (root level)")

    return base_path


def demonstrate_labeling():
    """Demonstrate the subdirectory labeling feature."""
    print("=" * 60)
    print("doc2lora Subdirectory Labeling Demonstration")
    print("=" * 60)

    # Create demo structure
    demo_path = create_demo_structure()

    try:
        # Parse the directory structure
        parser = DocumentParser()
        documents = parser.parse_directory(str(demo_path))

        print(f"\nParsed {len(documents)} documents:")
        print("-" * 60)

        # Group documents by label for better visualization
        by_label = {}
        for doc in documents:
            label = doc["label"]
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(doc)

        # Display results
        for label, docs in sorted(by_label.items()):
            print(f"\nLabel: '{label}'")
            print(f"Category Path: '{docs[0]['category_path']}'")
            print("Documents:")

            for doc in docs:
                print(f"  • {doc['filename']} ({doc['extension']})")
                print(f"    Content preview: {doc['content'][:50]}...")
                print(f"    Size: {doc['size']} bytes")

        # Show JSON output example
        print("\n" + "=" * 60)
        print("Example JSON output structure:")
        print("=" * 60)

        # Show one example document
        example_doc = documents[0]
        example_json = {
            "filename": example_doc["filename"],
            "label": example_doc["label"],
            "category_path": example_doc["category_path"],
            "content": example_doc["content"][:100] + "...",
            "extension": example_doc["extension"],
            "size": example_doc["size"],
        }

        print(json.dumps(example_json, indent=2))

        print("\n" + "=" * 60)
        print("Use Cases for Subdirectory Labeling:")
        print("=" * 60)
        print("• Organize training data by domain (legal, technical, marketing)")
        print("• Separate data by difficulty level (beginner, intermediate, advanced)")
        print("• Categorize by document type (contracts, manuals, FAQs)")
        print("• Group by language or region (en, es, fr)")
        print("• Organize by time period (2023, 2024, 2025)")

    finally:
        # Clean up
        import shutil

        shutil.rmtree(demo_path)
        print(f"\nCleaned up demo directory: {demo_path}")


if __name__ == "__main__":
    demonstrate_labeling()
