"""Command-line interface for doc2lora."""

import logging
from pathlib import Path

import click

from .core import convert

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """doc2lora: Convert documents to LoRA adapters for LLM fine-tuning."""
    pass


@cli.command()
@click.argument(
    "documents_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--output",
    "-o",
    default="lora_adapter.json",
    help="Output path for the LoRA adapter",
)
@click.option(
    "--model",
    "-m",
    default="microsoft/DialoGPT-small",
    help="Base model name for fine-tuning",
)
@click.option(
    "--max-length", default=512, help="Maximum sequence length for tokenization"
)
@click.option("--batch-size", default=4, help="Training batch size")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--learning-rate", default=5e-4, help="Learning rate for training")
@click.option("--lora-r", default=16, help="LoRA rank parameter")
@click.option("--lora-alpha", default=32, help="LoRA alpha parameter")
@click.option("--lora-dropout", default=0.1, help="LoRA dropout rate")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def convert_cmd(
    documents_path: str,
    output: str,
    model: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    verbose: bool,
):
    """Convert a folder of documents to LoRA adapter format."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        click.echo(f"Converting documents from: {documents_path}")
        click.echo(f"Output will be saved to: {output}")

        adapter_path = convert(
            documents_path=documents_path,
            output_path=output,
            model_name=model,
            max_length=max_length,
            batch_size=batch_size,
            num_epochs=epochs,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        click.echo(f"✅ LoRA adapter successfully created at: {adapter_path}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument(
    "documents_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def scan(documents_path: str):
    """Scan a directory for supported document files."""
    from .parsers import DocumentParser

    parser = DocumentParser()
    documents = parser.parse_directory(documents_path)

    click.echo(f"Found {len(documents)} supported documents:")

    for doc in documents:
        size_kb = doc["size"] / 1024
        click.echo(f"  📄 {doc['filename']} ({doc['extension']}, {size_kb:.1f} KB)")


@cli.command()
def formats():
    """List supported document formats."""
    from .parsers import DocumentParser

    click.echo("Supported document formats:")

    formats_info = [
        (".md", "Markdown files"),
        (".txt", "Text files"),
        (".pdf", "PDF documents"),
        (".html", "HTML files"),
        (".docx", "Word documents"),
        (".csv", "CSV files"),
        (".json", "JSON files"),
        (".yaml/.yml", "YAML files"),
        (".xml", "XML files"),
        (".tex", "LaTeX files"),
    ]

    for ext, description in formats_info:
        click.echo(f"  {ext:<12} {description}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
