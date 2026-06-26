"""Deploy LoRA adapters to Cloudflare Workers AI."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cloudflare BYO-LoRA limits (see developers.cloudflare.com/workers-ai/features/fine-tunes/loras)
CLOUDFLARE_MAX_RANK = 32
MAX_ADAPTER_BYTES = 300 * 1024 * 1024  # 300MB safetensors limit
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")

# default lora-capable base model endpoints by model_type (override as needed)
DEFAULT_CF_MODELS = {
    "mistral": "@cf/mistralai/mistral-7b-instruct-v0.2-lora",
    "gemma": "@cf/google/gemma-7b-it-lora",
    "llama": "@cf/meta-llama/llama-2-7b-chat-hf-lora",
    "qwen": "@cf/qwen/qwq-32b",
}


def resolve_adapter_dir(adapter_path: str) -> Path:
    """
    Resolve an adapter directory from a directory path or a .json metadata file.

    Args:
        adapter_path: Path to the adapter directory or the metadata JSON produced
            by ``save_adapter``

    Returns:
        Path to the directory holding adapter_config.json + adapter_model.safetensors
    """
    path = Path(adapter_path)

    if path.is_dir():
        return path

    if path.suffix == ".json" and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        adapter_dir = metadata.get("adapter_path")
        if adapter_dir:
            return Path(adapter_dir)

    raise ValueError(
        f"Could not resolve an adapter directory from: {adapter_path}. "
        "Pass the adapter directory or the metadata .json from save_adapter()."
    )


def validate_adapter(adapter_dir: str) -> Dict[str, Any]:
    """
    Validate that an adapter is Cloudflare Workers AI ready.

    Checks the required filenames, rank <= 32, presence of model_type, and the
    300MB safetensors size limit.

    Args:
        adapter_dir: Path to the adapter directory

    Returns:
        Dict with the resolved rank, model_type, and size in bytes

    Raises:
        ValueError: If the adapter is not Cloudflare-compatible
    """
    directory = Path(adapter_dir)
    if not directory.is_dir():
        raise ValueError(f"Adapter directory not found: {adapter_dir}")

    for name in REQUIRED_FILES:
        if not (directory / name).exists():
            raise ValueError(
                f"Adapter is missing required file '{name}'. Cloudflare Workers AI "
                f"requires exactly {REQUIRED_FILES}."
            )

    with open(directory / "adapter_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    rank = config.get("r")
    if rank is not None and rank > CLOUDFLARE_MAX_RANK:
        raise ValueError(
            f"LoRA rank {rank} exceeds the Cloudflare Workers AI limit of "
            f"{CLOUDFLARE_MAX_RANK}. Retrain with a lower --lora-r."
        )

    model_type = config.get("model_type")
    if not model_type:
        raise ValueError(
            "adapter_config.json is missing 'model_type' (mistral|gemma|llama|qwen). "
            "Re-run save_adapter() which injects it."
        )

    size_bytes = (directory / "adapter_model.safetensors").stat().st_size
    if size_bytes > MAX_ADAPTER_BYTES:
        raise ValueError(
            f"Adapter file is {size_bytes / 1024 / 1024:.0f}MB, over the Cloudflare "
            f"300MB limit. Lower the rank or trim target modules."
        )

    return {"rank": rank, "model_type": model_type, "size_bytes": size_bytes}


def deploy_adapter(
    adapter_path: str,
    finetune_name: str,
    cf_model: Optional[str] = None,
    backend: str = "wrangler",
    account_id: Optional[str] = None,
    api_token: Optional[str] = None,
) -> str:
    """
    Upload a LoRA adapter to Cloudflare Workers AI.

    Args:
        adapter_path: Adapter directory or the metadata .json from save_adapter()
        finetune_name: Name to register the finetune under (referenced via ``lora``)
        cf_model: Lora-capable base model endpoint; derived from model_type if omitted
        backend: "wrangler" (shells out to the CLI) or "rest" (direct API upload)
        account_id: Cloudflare account id (REST backend; or CLOUDFLARE_ACCOUNT_ID)
        api_token: Cloudflare API token (REST backend; or CLOUDFLARE_API_TOKEN)

    Returns:
        The finetune name (and id, for the REST backend)
    """
    adapter_dir = resolve_adapter_dir(adapter_path)
    info = validate_adapter(adapter_dir)

    if cf_model is None:
        cf_model = DEFAULT_CF_MODELS.get(info["model_type"])
        if cf_model is None:
            raise ValueError(
                f"No default Cloudflare model for model_type '{info['model_type']}'. "
                "Pass cf_model explicitly."
            )
        logger.warning(
            f"ℹ️  Using default Cloudflare model '{cf_model}'; verify it against the "
            "live models list (capabilities=LoRA)."
        )

    logger.info(
        f"🚀 Deploying adapter '{finetune_name}' (rank {info['rank']}, "
        f"{info['size_bytes'] / 1024 / 1024:.1f}MB) to {cf_model} via {backend}"
    )

    if backend == "wrangler":
        return _deploy_wrangler(adapter_dir, finetune_name, cf_model)
    if backend == "rest":
        return _deploy_rest(adapter_dir, finetune_name, cf_model, account_id, api_token)
    raise ValueError(f"Unknown backend '{backend}' (expected 'wrangler' or 'rest')")


def _deploy_wrangler(adapter_dir: Path, finetune_name: str, cf_model: str) -> str:
    """Deploy via the wrangler CLI: wrangler ai finetune create <model> <name> <dir>."""
    if shutil.which("wrangler") is None:
        raise RuntimeError(
            "wrangler CLI not found. Install it (npm install -g wrangler) or use the "
            "REST backend."
        )

    cmd = [
        "wrangler",
        "ai",
        "finetune",
        "create",
        cf_model,
        finetune_name,
        str(adapter_dir),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"✅ Finetune '{finetune_name}' created on Cloudflare Workers AI")
    return finetune_name


def _deploy_rest(
    adapter_dir: Path,
    finetune_name: str,
    cf_model: str,
    account_id: Optional[str],
    api_token: Optional[str],
) -> str:
    """Deploy via the Cloudflare REST API (multipart upload of both files)."""
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The REST backend needs 'requests' (installed with transformers). "
            "Install it or use the wrangler backend."
        )

    account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = api_token or os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise ValueError(
            "REST backend needs account_id + api_token (or CLOUDFLARE_ACCOUNT_ID and "
            "CLOUDFLARE_API_TOKEN env vars)."
        )

    base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/finetunes"
    headers = {"Authorization": f"Bearer {api_token}"}

    create = requests.post(
        base,
        headers=headers,
        json={"model": cf_model, "name": finetune_name, "description": "doc2lora"},
    )
    create.raise_for_status()
    finetune_id = create.json()["result"]["id"]
    logger.info(f"Created finetune id: {finetune_id}")

    for name in REQUIRED_FILES:
        with open(adapter_dir / name, "rb") as fh:
            upload = requests.post(
                f"{base}/{finetune_id}/finetune-assets",
                headers=headers,
                data={"file_name": name},
                files={"file": (name, fh)},
            )
        upload.raise_for_status()
        logger.info(f"Uploaded {name}")

    logger.info(f"✅ Finetune '{finetune_name}' ({finetune_id}) uploaded")
    return f"{finetune_name} ({finetune_id})"
