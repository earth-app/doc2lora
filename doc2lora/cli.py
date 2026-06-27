"""Command-line interface for doc2lora."""

import logging
import os
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .core import convert, convert_from_r2

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@click.group()
@click.version_option(version=__version__)
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
    "--lora-r",
    default=8,
    help="LoRA rank parameter (Cloudflare Workers AI supports up to 32)",
)
@click.option("--lora-alpha", default=16, help="LoRA alpha parameter")
@click.option("--lora-dropout", default=0.1, help="LoRA dropout rate")
@click.option(
    "--gradient-accumulation-steps",
    default=1,
    type=int,
    help="Accumulate grads to emulate a larger batch on low-memory machines",
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=True,
    help="Trade compute for memory (helps on low-RAM machines)",
)
@click.option(
    "--load-in-4bit",
    is_flag=True,
    default=False,
    help="Use 4-bit QLoRA (requires bitsandbytes + CUDA)",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cuda", "mps", "cpu", "auto"], case_sensitive=False),
    help="Device to use for training (auto-detects by default)",
)
@click.option(
    "--attn-implementation",
    default=None,
    type=click.Choice(["sdpa", "flash_attention_2", "eager"], case_sensitive=False),
    help="Attention kernel (sdpa auto-picked; flash_attention_2 is CUDA-only)",
)
@click.option(
    "--group-by-length/--no-group-by-length",
    default=None,
    help="Length-grouped batching (auto-on at batch>=2; force on/off here)",
)
@click.option(
    "--dataloader-num-workers",
    default=None,
    type=int,
    help="DataLoader worker processes (default in-process; raise on Linux/CUDA)",
)
@click.option(
    "--torch-compile/--no-torch-compile",
    default=None,
    help="torch.compile: auto-on for CUDA + large corpora; force on/off here",
)
@click.option(
    "--optim",
    default=None,
    help="Override optimizer (default: fused AdamW on CUDA, else adamw_torch)",
)
@click.option(
    "--audio-backend",
    default="faster-whisper",
    type=click.Choice(
        ["faster-whisper", "openai-whisper", "speech_recognition", "auto"],
        case_sensitive=False,
    ),
    help="Speech-to-text backend for audio/video (default: faster-whisper)",
)
@click.option(
    "--whisper-model",
    default="base",
    help="Whisper model size for the whisper backends (tiny..large-v3)",
)
@click.option(
    "--ocr-languages",
    default="eng",
    help="Tesseract language code(s) for image/video OCR (e.g. eng, eng+fra)",
)
@click.option(
    "--video-frame-interval",
    default=1.0,
    type=float,
    help="Seconds between sampled video frames for on-screen-text OCR",
)
@click.option(
    "--max-workers",
    default=None,
    type=int,
    help="Thread-pool size for parsing the documents directory (default: auto)",
)
@click.option(
    "--chunk/--no-chunk",
    default=True,
    help="Split docs longer than --max-length into multiple examples vs truncate",
)
@click.option(
    "--chunk-overlap",
    default=0,
    type=int,
    help="Token overlap between consecutive chunks (default 0)",
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
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    load_in_4bit: bool,
    device: Optional[str],
    attn_implementation: Optional[str],
    group_by_length: Optional[bool],
    dataloader_num_workers: Optional[int],
    torch_compile: Optional[bool],
    optim: Optional[str],
    audio_backend: str,
    whisper_model: str,
    ocr_languages: str,
    video_frame_interval: float,
    max_workers: Optional[int],
    chunk: bool,
    chunk_overlap: int,
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
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            load_in_4bit=load_in_4bit,
            device=None if device == "auto" else device,
            attn_implementation=attn_implementation,
            group_by_length=group_by_length,
            dataloader_num_workers=dataloader_num_workers,
            torch_compile=torch_compile,
            optim=optim,
            audio_backend=audio_backend,
            whisper_model_size=whisper_model,
            ocr_languages=ocr_languages,
            video_frame_interval=video_frame_interval,
            max_workers=max_workers,
            chunk_long_documents=chunk,
            chunk_overlap=chunk_overlap,
        )

        click.echo(f"✅ LoRA adapter successfully created at: {adapter_path}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument(
    "documents_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cuda", "mps", "cpu", "auto"], case_sensitive=False),
    help="Device to assume for the training-time estimate",
)
def scan(documents_path: str, device: Optional[str]):
    """Scan a directory for supported document files."""
    from .parsers import DocumentParser
    from .utils import create_training_summary, estimate_training_time

    resolved_device = None if device == "auto" else device
    parser = DocumentParser()
    documents = parser.parse_directory(documents_path)

    click.echo(f"Found {len(documents)} supported documents:")

    for doc in documents:
        size_kb = doc["size"] / 1024
        # per-file estimate is based on the extracted text (what actually trains)
        content_mb = len(doc.get("content", "")) / (1024 * 1024)
        per_file = estimate_training_time(1, content_mb, device=resolved_device)
        click.echo(
            f"  📄 {doc['filename']} ({doc['extension']}, {size_kb:.1f} KB, "
            f"est. ~{per_file})"
        )

    if documents:
        summary = create_training_summary(documents, device=resolved_device)
        click.echo(
            f"\nTotal size: {summary['total_size_formatted']} "
            f"across {len(summary['file_types'])} file type(s)"
        )
        click.echo(
            f"Estimated training time (~small model): "
            f"{summary['estimated_training_time']}"
        )
        click.echo(
            "Note: rough estimate from extracted text; 7B-class models are ~20-40x "
            "slower (QLoRA on CUDA recovers much of that)."
        )


@cli.command()
def formats():
    """List supported document formats."""
    from .parsers import DocumentParser

    click.echo("Supported document formats:")

    formats_info = [
        (".md/.rst", "Markdown / reStructuredText"),
        (".txt", "Text files"),
        (".pdf", "PDF documents"),
        (".html", "HTML files"),
        (".docx", "Word documents"),
        (".pptx", "PowerPoint slides (text + notes)"),
        (".odt/.ods", "OpenDocument text / spreadsheet"),
        (".rtf", "Rich Text Format"),
        (".epub", "EPUB e-books"),
        (".csv", "CSV files"),
        (".json", "JSON files"),
        (".ipynb", "Jupyter notebooks (markdown + code)"),
        (".yaml/.yml", "YAML files"),
        (".xml", "XML files"),
        (".tex", "LaTeX files"),
        (
            "audio",
            "Speech-to-text via whisper (.wav, .mp3, .m4a, .flac, .aac, .ogg, ...)",
        ),
        (
            "images",
            "OCR text recognition (.png, .jpg, .bmp, .gif, .tiff, .webp, ...)",
        ),
        (".svg", "Vector image text extracted from markup (no OCR)"),
        (
            "video",
            "Audio transcript + on-screen-text OCR (.mp4, .avi, .mov, .mkv, ...)",
        ),
        (
            "source code",
            "Read as plaintext (.py, .js, .rs, .kt, .c/.cpp, .go, .dart, ...)",
        ),
        (".zip", "ZIP archives containing supported documents"),
        (".tar", "TAR archives containing supported documents"),
        (".tar.gz/.tgz", "Gzip-compressed TAR archives"),
        (".tar.bz2/.tbz2", "Bzip2-compressed TAR archives"),
        (".tar.xz/.txz", "XZ-compressed TAR archives"),
        (".7z", "7-Zip archives"),
        (".gz/.bz2/.xz", "Single-file compressed documents"),
    ]

    for ext, description in formats_info:
        click.echo(f"  {ext:<15} {description}")

    code_exts = ", ".join(sorted(DocumentParser.CODE_EXTENSIONS))
    click.echo(f"\nSource-code extensions read as plaintext: {code_exts}")
    click.echo("\nNote: Archive formats (.zip, .tar, .7z, etc.) will extract and parse")
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
    "--lora-r",
    default=8,
    help="LoRA rank parameter (Cloudflare Workers AI supports up to 32)",
)
@click.option("--lora-alpha", default=16, help="LoRA alpha parameter")
@click.option("--lora-dropout", default=0.1, help="LoRA dropout rate")
@click.option(
    "--gradient-accumulation-steps",
    default=1,
    type=int,
    help="Accumulate grads to emulate a larger batch on low-memory machines",
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=True,
    help="Trade compute for memory (helps on low-RAM machines)",
)
@click.option(
    "--load-in-4bit",
    is_flag=True,
    default=False,
    help="Use 4-bit QLoRA (requires bitsandbytes + CUDA)",
)
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
@click.option(
    "--attn-implementation",
    default=None,
    type=click.Choice(["sdpa", "flash_attention_2", "eager"], case_sensitive=False),
    help="Attention kernel (sdpa auto-picked; flash_attention_2 is CUDA-only)",
)
@click.option(
    "--group-by-length/--no-group-by-length",
    default=None,
    help="Length-grouped batching (auto-on at batch>=2; force on/off here)",
)
@click.option(
    "--dataloader-num-workers",
    default=None,
    type=int,
    help="DataLoader worker processes (default in-process; raise on Linux/CUDA)",
)
@click.option(
    "--torch-compile/--no-torch-compile",
    default=None,
    help="torch.compile: auto-on for CUDA + large corpora; force on/off here",
)
@click.option(
    "--optim",
    default=None,
    help="Override optimizer (default: fused AdamW on CUDA, else adamw_torch)",
)
@click.option(
    "--audio-backend",
    default="faster-whisper",
    type=click.Choice(
        ["faster-whisper", "openai-whisper", "speech_recognition", "auto"],
        case_sensitive=False,
    ),
    help="Speech-to-text backend for audio/video (default: faster-whisper)",
)
@click.option(
    "--whisper-model",
    default="base",
    help="Whisper model size for the whisper backends (tiny..large-v3)",
)
@click.option(
    "--ocr-languages",
    default="eng",
    help="Tesseract language code(s) for image/video OCR (e.g. eng, eng+fra)",
)
@click.option(
    "--video-frame-interval",
    default=1.0,
    type=float,
    help="Seconds between sampled video frames for on-screen-text OCR",
)
@click.option(
    "--max-workers",
    default=None,
    type=int,
    help="Thread-pool size for parsing the downloaded files (default: auto)",
)
@click.option(
    "--chunk/--no-chunk",
    default=True,
    help="Split docs longer than --max-length into multiple examples vs truncate",
)
@click.option(
    "--chunk-overlap",
    default=0,
    type=int,
    help="Token overlap between consecutive chunks (default 0)",
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
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    load_in_4bit: bool,
    device: Optional[str],
    r2_access_key_id: str,
    r2_secret_access_key: str,
    endpoint_url: str,
    region_name: str,
    env_file: str,
    attn_implementation: Optional[str],
    group_by_length: Optional[bool],
    dataloader_num_workers: Optional[int],
    torch_compile: Optional[bool],
    optim: Optional[str],
    audio_backend: str,
    whisper_model: str,
    ocr_languages: str,
    video_frame_interval: float,
    max_workers: Optional[int],
    chunk: bool,
    chunk_overlap: int,
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
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            load_in_4bit=load_in_4bit,
            device=None if device == "auto" else device,
            attn_implementation=attn_implementation,
            group_by_length=group_by_length,
            dataloader_num_workers=dataloader_num_workers,
            torch_compile=torch_compile,
            optim=optim,
            audio_backend=audio_backend,
            whisper_model_size=whisper_model,
            ocr_languages=ocr_languages,
            video_frame_interval=video_frame_interval,
            max_workers=max_workers,
            chunk_long_documents=chunk,
            chunk_overlap=chunk_overlap,
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


@cli.command()
@click.argument("adapter_path", type=click.Path(exists=True))
@click.argument("finetune_name")
@click.option(
    "--cf-model",
    default=None,
    help="Lora-capable base model endpoint (derived from model_type if omitted)",
)
@click.option(
    "--backend",
    type=click.Choice(["wrangler", "rest"], case_sensitive=False),
    default="wrangler",
    help="Upload via the wrangler CLI or the Cloudflare REST API",
)
@click.option(
    "--account-id",
    envvar="CLOUDFLARE_ACCOUNT_ID",
    help="Cloudflare account id (REST backend; or CLOUDFLARE_ACCOUNT_ID)",
)
@click.option(
    "--api-token",
    envvar="CLOUDFLARE_API_TOKEN",
    help="Cloudflare API token (REST backend; or CLOUDFLARE_API_TOKEN)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def deploy(
    adapter_path: str,
    finetune_name: str,
    cf_model: Optional[str],
    backend: str,
    account_id: Optional[str],
    api_token: Optional[str],
    verbose: bool,
):
    """Upload a trained adapter to Cloudflare Workers AI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .deploy import deploy_adapter

    try:
        result = deploy_adapter(
            adapter_path=adapter_path,
            finetune_name=finetune_name,
            cf_model=cf_model,
            backend=backend.lower(),
            account_id=account_id,
            api_token=api_token,
        )
        click.echo(f"✅ Deployed: {result}")
        click.echo(
            f'   Reference it at inference with the lora param: "{finetune_name}"'
        )
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
