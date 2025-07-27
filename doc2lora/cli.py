"""Command-line interface for doc2lora."""

import logging
import os
from pathlib import Path
from typing import Optional

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
@click.option(
    "--max-steps",
    default=None,
    type=int,
    help="Maximum number of training steps (overrides epochs if set)",
)
@click.option("--learning-rate", default=5e-4, help="Learning rate for training")
@click.option(
    "--lora-r", default=8, help="LoRA rank parameter (max 8 for Cloudflare Workers AI)"
)
@click.option("--lora-alpha", default=16, help="LoRA alpha parameter")
@click.option("--lora-dropout", default=0.1, help="LoRA dropout rate")
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cuda", "mps", "cpu", "auto"], case_sensitive=False),
    help="Device to use for training (auto-detects by default)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def convert_cmd(
    documents_path: str,
    output: str,
    model: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    max_steps: Optional[int],
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: Optional[str],
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
            num_epochs=epochs if max_steps is None else None,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=None if device == "auto" else device,
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
        (".zip", "ZIP archives containing supported documents"),
        (".tar", "TAR archives containing supported documents"),
        (".tar.gz/.tgz", "Gzip-compressed TAR archives"),
        (".tar.bz2/.tbz2", "Bzip2-compressed TAR archives"),
        (".tar.xz/.txz", "XZ-compressed TAR archives"),
    ]

    for ext, description in formats_info:
        click.echo(f"  {ext:<15} {description}")

    click.echo("\nNote: Archive formats (.zip, .tar, etc.) will extract and parse")
    click.echo("      any supported document files they contain.")


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
@click.option(
    "--max-steps",
    default=None,
    type=int,
    help="Maximum number of training steps (overrides epochs if set)",
)
@click.option("--learning-rate", default=5e-4, help="Learning rate for training")
@click.option(
    "--lora-r", default=8, help="LoRA rank parameter (max 8 for Cloudflare Workers AI)"
)
@click.option("--lora-alpha", default=16, help="LoRA alpha parameter")
@click.option("--lora-dropout", default=0.1, help="LoRA dropout rate")
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cuda", "mps", "cpu", "auto"], case_sensitive=False),
    help="Device to use for training (auto-detects by default)",
)
@click.option(
    "--r2-access-key-id",
    envvar="R2_ACCESS_KEY_ID",
    help="R2 access key ID (can also be set via R2_ACCESS_KEY_ID env var)",
)
@click.option(
    "--r2-secret-access-key",
    envvar="R2_SECRET_ACCESS_KEY",
    help="R2 secret access key (can also be set via R2_SECRET_ACCESS_KEY env var)",
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
@click.option(
    "--env-file",
    help="Path to .env file containing R2 credentials",
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
    max_steps: Optional[int],
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: Optional[str],
    r2_access_key_id: str,
    r2_secret_access_key: str,
    endpoint_url: str,
    region_name: str,
    env_file: str,
    verbose: bool,
):
    """Convert documents from an R2 bucket to LoRA adapter format."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load .env file if provided
    if env_file:
        from .utils import load_env_file

        try:
            load_env_file(env_file)
            click.echo(f"Loaded credentials from: {env_file}")
        except FileNotFoundError as e:
            click.echo(f"❌ Error: {e}", err=True)
            raise click.Abort()

    # If credentials not provided directly, try to get from environment
    # (which may have been loaded from .env file)
    if not r2_access_key_id:
        r2_access_key_id = os.getenv("R2_ACCESS_KEY_ID") or os.getenv(
            "AWS_ACCESS_KEY_ID"
        )
    if not r2_secret_access_key:
        r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
    if not endpoint_url:
        endpoint_url = os.getenv("R2_ENDPOINT_URL")

    # Validate required credentials
    if not r2_access_key_id or not r2_secret_access_key:
        click.echo(
            "❌ Error: R2 credentials are required. Provide them via:\n"
            "  --r2-access-key-id and --r2-secret-access-key options, or\n"
            "  R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables, or\n"
            "  AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables (legacy), or\n"
            "  --env-file option pointing to a .env file",
            err=True,
        )
        raise click.Abort()

    if not endpoint_url:
        click.echo(
            "❌ Error: R2 endpoint URL is required. Provide it via:\n"
            "  --endpoint-url option, or\n"
            "  R2_ENDPOINT_URL environment variable, or\n"
            "  --env-file option pointing to a .env file\n"
            "  Example: https://your-account.r2.cloudflarestorage.com",
            err=True,
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
            num_epochs=epochs if max_steps is None else None,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=None if device == "auto" else device,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
            env_file=env_file,
        )

        click.echo(f"✅ LoRA adapter successfully created at: {adapter_path}")

    except Exception as e:
        if "No files found" in str(e):
            click.echo(f"❌ Error: {e}", err=True)
            click.echo("\n💡 Troubleshooting tips:", err=True)
            click.echo("  • Check that your bucket contains files", err=True)
            click.echo(
                "  • Verify the folder prefix (if specified) is correct", err=True
            )
            click.echo(
                "  • Ensure files are in supported formats (.md, .txt, .pdf, etc.)",
                err=True,
            )
        elif "Bucket" in str(e) and "does not exist" in str(e):
            click.echo(f"❌ Error: {e}", err=True)
            click.echo("\n💡 Troubleshooting tips:", err=True)
            click.echo("  • Check the bucket name is correct", err=True)
            click.echo("  • Verify the bucket exists in your R2 account", err=True)
            click.echo(
                "  • Ensure your credentials have access to this bucket", err=True
            )
        elif "endpoint" in str(e).lower():
            click.echo(f"❌ Error: {e}", err=True)
            click.echo("\n💡 Troubleshooting tips:", err=True)
            click.echo(
                "  • R2 endpoint format: https://your-account-id.r2.cloudflarestorage.com",
                err=True,
            )
            click.echo(
                "  • Do NOT include the bucket name in the endpoint URL", err=True
            )
            click.echo(
                "  • Get your endpoint from Cloudflare dashboard > R2 > Manage R2 API tokens",
                err=True,
            )
        else:
            click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
