"""Command-line interface for doc2lora."""

import logging
from pathlib import Path

import click

from .core import convert, convert_from_r2

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


@cli.command()
@click.argument("bucket_name")
@click.option(
    "--folder-prefix",
    "-f",
    help="Optional folder prefix within the bucket",
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
@click.option(
    "--aws-access-key-id",
    envvar="AWS_ACCESS_KEY_ID",
    help="AWS access key ID for R2 (can also be set via AWS_ACCESS_KEY_ID env var)",
)
@click.option(
    "--aws-secret-access-key",
    envvar="AWS_SECRET_ACCESS_KEY",
    help="AWS secret access key for R2 (can also be set via AWS_SECRET_ACCESS_KEY env var)",
)
@click.option(
    "--endpoint-url",
    envvar="R2_ENDPOINT_URL",
    help="R2 endpoint URL (can also be set via R2_ENDPOINT_URL env var)",
)
@click.option(
    "--region-name",
    default="auto",
    help="Region name (default: auto for R2)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def convert_r2(
    bucket_name: str,
    folder_prefix: str,
    output: str,
    model: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    endpoint_url: str,
    region_name: str,
    verbose: bool,
):
    """Convert documents from an R2 bucket to LoRA adapter format."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate required credentials
    if not aws_access_key_id or not aws_secret_access_key:
        click.echo(
            "❌ Error: AWS credentials are required. Provide them via:\n"
            "  --aws-access-key-id and --aws-secret-access-key options, or\n"
            "  AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
            err=True
        )
        raise click.Abort()

    if not endpoint_url:
        click.echo(
            "❌ Error: R2 endpoint URL is required. Provide it via:\n"
            "  --endpoint-url option, or\n"
            "  R2_ENDPOINT_URL environment variable\n"
            "  Example: https://your-account.r2.cloudflarestorage.com",
            err=True
        )
        raise click.Abort()

    try:
        click.echo(f"Converting documents from R2 bucket: {bucket_name}")
        if folder_prefix:
            click.echo(f"Folder prefix: {folder_prefix}")
        click.echo(f"Output will be saved to: {output}")

        adapter_path = convert_from_r2(
            bucket_name=bucket_name,
            folder_prefix=folder_prefix,
            output_path=output,
            model_name=model,
            max_length=max_length,
            batch_size=batch_size,
            num_epochs=epochs,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

        click.echo(f"✅ LoRA adapter successfully created at: {adapter_path}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
