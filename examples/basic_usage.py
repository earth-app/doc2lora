"""Example usage of doc2lora library."""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import doc2lora
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_document_parsing():
    """Demonstrate document parsing functionality that works without ML dependencies."""

    print("=== doc2lora Document Parsing Demo ===")

    # Get the path to our example documents
    docs_path = Path(__file__).parent / "example_documents"

    if not docs_path.exists():
        print(f"❌ Example documents directory not found: {docs_path}")
        print(
            "   Please ensure the example_documents directory exists with sample files."
        )
        return

    print(f"📁 Scanning directory: {docs_path}")

    # Manually parse documents to demonstrate the core functionality
    supported_extensions = {
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
    documents = []

    for file_path in docs_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                # Read the file content based on extension
                if file_path.suffix.lower() in {
                    ".txt",
                    ".md",
                    ".json",
                    ".yaml",
                    ".yml",
                    ".xml",
                    ".tex",
                    ".csv",
                }:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                else:
                    # For binary files like PDF, DOCX, just indicate they would be processed
                    content = f"[Binary file: {file_path.name} - would be processed by appropriate parser]"

                doc_info = {
                    "content": content,
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "extension": file_path.suffix.lower(),
                    "size": file_path.stat().st_size,
                }
                documents.append(doc_info)

            except Exception as e:
                print(f"⚠️  Could not parse {file_path.name}: {e}")

    print(f"\n✅ Successfully found and parsed {len(documents)} documents:")

    total_size = 0
    for doc in documents:
        size_kb = doc["size"] / 1024
        total_size += doc["size"]

        # Show file info
        print(f"\n📄 {doc['filename']} ({doc['extension']})")
        print(f"   Size: {size_kb:.1f} KB")

        # Show content preview
        content_preview = doc["content"].strip()[:150].replace("\n", " ")
        print(f"   Preview: {content_preview}...")

        # Show what this would contribute to training
        word_count = len(doc["content"].split())
        print(f"   Words: ~{word_count} words")

    total_size_kb = total_size / 1024
    total_words = sum(len(doc["content"].split()) for doc in documents)

    print(f"\n📊 Summary:")
    print(f"   Total documents: {len(documents)}")
    print(f"   Total size: {total_size_kb:.1f} KB")
    print(f"   Total words: ~{total_words} words")
    print(f"   File types: {set(doc['extension'] for doc in documents)}")

    return documents


def demo_lora_conversion_info():
    """Show what the LoRA conversion would do (without actually doing it)."""

    print(f"\n{'='*50}")
    print("🚀 LoRA Conversion Process (requires ML dependencies)")
    print(f"{'='*50}")

    print("\n📋 What the conversion process would do:")
    print("   1. Parse all documents in the directory")
    print("   2. Tokenize the text content for the language model")
    print("   3. Create training datasets from the documents")
    print("   4. Initialize a base language model (e.g., DialoGPT)")
    print("   5. Apply LoRA (Low-Rank Adaptation) configuration")
    print("   6. Fine-tune the model on your document content")
    print("   7. Save the LoRA adapter weights")

    print("\n⚙️  Typical LoRA parameters:")
    print("   - Rank (r): 16 (controls adaptation capacity)")
    print("   - Alpha: 32 (controls adaptation strength)")
    print("   - Dropout: 0.1 (prevents overfitting)")
    print("   - Target modules: attention layers")

    print("\n💾 Output format:")
    print("   - JSON file with adapter metadata")
    print("   - Adapter weights directory")
    print("   - Compatible with Hugging Face PEFT")

    print("\n🔧 To enable full functionality, install:")
    print("   pip install torch transformers peft datasets")
    print(
        "   pip install PyPDF2 python-docx beautifulsoup4 PyYAML  # for additional formats"
    )


def main():
    """Main demo function."""

    try:
        # Demo the document parsing (this works without ML dependencies)
        documents = demo_document_parsing()

        # Show information about what LoRA conversion would do
        demo_lora_conversion_info()

        # Try to actually run the conversion if dependencies are available
        print(f"\n{'='*50}")
        print("🧪 Attempting actual LoRA conversion...")
        print(f"{'='*50}")

        try:
            from doc2lora import convert

            print("✅ ML dependencies found! Running actual conversion...")

            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)

            docs_path = Path(__file__).parent / "example_documents"

            # Run with minimal settings for demo
            adapter_path = convert(
                documents_path=str(docs_path),
                output_path=str(output_dir / "demo_adapter.json"),
                num_epochs=1,  # Quick demo
                batch_size=1,
                max_length=256,  # Shorter sequences for demo
            )
            print(f"🎉 LoRA adapter successfully created at: {adapter_path}")

        except ImportError as e:
            print(f"⚠️  ML dependencies not available: {e}")
            print(
                "   Document parsing works, but LoRA training requires additional packages."
            )
            print("   Run: pip install torch transformers peft datasets")
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            print(
                "   This might be due to insufficient resources or model download issues."
            )

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
