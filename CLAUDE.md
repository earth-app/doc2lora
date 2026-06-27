## CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install for development (editable, with dev extras):

```bash
pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest tests/ -v --cov=doc2lora --cov-report=xml
```

Run a single test (module / class / case):

```bash
pytest tests/test_parsers.py -v
pytest tests/test_core.py::TestCore::test_convert_basic -v
```

Lint and format (the CI gates that must pass on every push/PR):

```bash
flake8 doc2lora/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 doc2lora/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
black --check doc2lora/ tests/ examples/
isort --check-only doc2lora/ tests/ examples/
```

Auto-fix formatting before committing:

```bash
black doc2lora/ tests/ examples/
isort doc2lora/ tests/ examples/
```

Build & publish (release flow runs in CI on `release: published`):

```bash
python -m build
python -m twine check dist/*
```

CLI smoke test (after `pip install -e .`):

```bash
doc2lora formats                              # list supported extensions
doc2lora scan path/to/docs                    # parse-only, no training
doc2lora convert path/to/docs -o adapter.json # full training run
doc2lora convert-r2 my-bucket --env-file .env -o adapter.json
```

CI matrix is Python 3.11-3.14 on `actions/setup-python@v5` with pip cache; the format/build jobs run on 3.14. Match those versions locally when reproducing CI failures. Publishing to PyPI is owned by `release.yml` (Trusted Publishing on `release: published`); `build.yml` only tests and builds artifacts.

## Architecture

The package has a three-layer pipeline. Entry points `convert()` and `convert_from_r2()` in `doc2lora/core.py` orchestrate it; everything else is a layer they call into.

**Lazy heavy imports (keep this):** `core.py` imports `LoRATrainer` lazily (inside `convert()`) and `utils.py` imports `boto3` lazily (inside `download_from_r2_bucket`), so `scan` / `formats` / `deploy` / `--version` and any plain `import doc2lora` never pull in torch/transformers/peft/accelerate or boto3. Do NOT re-add eager `from .lora_trainer import ...` / `import boto3` at module top - it makes parse-only commands crash in minimal/broken environments (v1.0.2 fix).

1. **Ingestion** (`doc2lora/parsers.py`, `doc2lora/utils.py`)
   - `DocumentParser` walks a directory, dispatches per extension to a `_parse_<ext>` method, and returns a list of `{content, filename, filepath, extension, size, label, category_path}` dicts.
   - **Subdirectory labeling**: every document's `label` is derived from `<subdir>_<filename-stem>` (root files become `root_<stem>`). This is a contract consumed downstream; preserve it when changing parsers.
   - **Archive handling**: `.zip` / `.tar(.gz|.bz2|.xz)` are extracted to a temp dir, the same per-extension dispatch runs on the contents, then the temp dir is cleaned up. Only extensions in `DOCUMENT_EXTENSIONS` are pulled out of archives; everything else is skipped.
   - **R2 ingestion**: `download_from_r2_bucket()` in `utils.py` is a thin boto3 shim that downloads a bucket (optionally filtered by `folder_prefix`) into a temp dir, then `convert_from_r2()` delegates to `convert(documents_path=temp_dir, ...)` and cleans up in `finally`. Credentials resolve from explicit args -> `.env` file -> `R2_*` / `AWS_*` env vars.
   - Three core inputs are equivalent: a folder path (`documents_path=`), or in-memory content (`input_data=`) as `str`, `List[str]`, `bytes`, or `List[bytes]`. In-memory inputs synthesize fake `memory://input_document_<i>.txt` filenames and skip parser dispatch entirely.
   - **Media formats (v1.0.1)**: raster images go through OCR (`pytesseract`); `.svg` text is read from the markup (no OCR); audio is transcribed via Whisper (`faster-whisper` default -> `openai-whisper` -> `speech_recognition`, resolved in `_resolve_audio_backend`); video combines a Whisper transcript with per-frame OCR deduped by identical on-screen text. `parse_directory(max_workers=)` parses files concurrently with a thread pool and returns them sorted by path for stable order. Whisper models are lazily loaded once behind a lock.

2. **Training** (`doc2lora/lora_trainer.py`)
   - `LoRATrainer` loads a HuggingFace model (`AutoModelForCausalLM`) + tokenizer, attaches a PEFT `LoraConfig`, and trains with the HF `Trainer`.
   - `get_device()` auto-picks `cuda` -> `mps` -> `cpu`; pass `device=` to override. Precision: bf16 on bf16-capable CUDA and Apple MPS (macOS 14+, `_mps_supports_bf16`), fp16 on other CUDA GPUs, fp32 on CPU and older MPS (fp16 autocast on MPS is NaN-prone, so it is never auto-enabled). CUDA OOM (and other load failures) fall back to CPU/fp32 at `_load_model()`. `apply_runtime_perf_flags()` enables TF32 matmul (Ampere+; no-op elsewhere). `_build_training_arguments()` sets fused AdamW on CUDA (torch>=2.8), CUDA-gated `dataloader_pin_memory`, `pad_to_multiple_of=8`, and only passes version-sensitive fields the installed transformers accepts (e.g. `group_by_length` was renamed `train_sampling_strategy` in transformers 5.x).
   - HuggingFace auth: `HF_API_KEY` or `HUGGINGFACE_API_TOKEN` env vars are read and threaded into `from_pretrained(token=...)` so gated models (Mistral, Llama, Gemma) work transparently.
   - `save_adapter(output_path)` writes a single JSON file that bundles the PEFT adapter weights and metadata in a format compatible with Cloudflare Workers AI.
   - **Rank constraint**: Cloudflare Workers AI accepts BYO LoRA adapters up to **rank 32** (300MB safetensors limit). `lora_r > 32` warns; 9-32 logs an info note. The CLI default stays at **8** for broadest compatibility - keep it at 8 unless the demo's deployment story changes. The ceiling is `CLOUDFLARE_MAX_RANK` in both `lora_trainer.py` and `deploy.py`.
   - **Low-resource training**: gradient checkpointing is on by default (`use_reentrant=False`); `gradient_accumulation_steps` and `load_in_4bit` (QLoRA; CUDA + bitsandbytes only, else warn-and-fallback) thread through `convert()` / `convert_from_r2()` / the CLI. **Opt-in / auto speedups** thread through too: `group_by_length` (None=auto: on at batch_size>=2 via `_should_group_by_length`; True/False force), `torch_compile` (None=auto: on for CUDA + corpora >= ~10 MB text via `_should_compile`; True/False force), `attn_implementation` (sdpa default; flash_attention_2 falls back to eager on failure), `optim`, and `dataloader_num_workers`.
   - **Document chunking**: `_prepare_dataset` auto-chunks documents longer than `max_length` into multiple training examples (`chunk_long_documents`, default on; `chunk_overlap` for window overlap; `--no-chunk` reverts to truncating each doc to its first `max_length` tokens). Training time therefore scales with total tokens, not document count.

3. **Deploy** (`doc2lora/deploy.py`)
   - `deploy_adapter()` validates an adapter (required filenames, rank <= 32, `model_type` set, < 300MB) then uploads via the `wrangler` CLI (default) or the Cloudflare REST API. Exposed as `doc2lora deploy`.

4. **CLI** (`doc2lora/cli.py`)
   - `click` group with commands: `convert`, `convert-r2`, `scan`, `formats`, `deploy`. `convert*` forward to `core.convert(*)`; `scan` / `formats` touch `DocumentParser`; `deploy` calls `deploy.deploy_adapter`.
   - `--device auto` is translated to `device=None` before reaching `core.convert` (the sentinel for auto-detect).

### Packaging & dependencies (v1.0)
   - Version is single-sourced from `doc2lora/__init__.py` (`pyproject.toml` reads it via `dynamic = ["version"]`).
   - Core `dependencies` are runtime-only (torch/transformers/peft/datasets/click/tqdm/numpy/pandas). Format/integration deps live in **extras**: `[docs]`, `[image]`, `[audio]`, `[video]`, `[r2]`, `[quant]`, `[all]`, `[dev]`. `requirements.txt` is the full install. Keep the optional-dependency guard pattern; don't promote extras to core or module-level imports.

### Optional-dependency pattern

`parsers.py` imports `pypdf`, `python-docx`, `beautifulsoup4`, `PyYAML`, `openpyxl`, `pytesseract`/`Pillow`, `opencv-python` (`cv2`), and the Whisper backends (`faster-whisper`, `openai-whisper`) inside `try/except ImportError`, then logs a one-time warning at `DocumentParser.__init__` listing what's missing. (ODT/ODS and SVG need no optional dep - they're parsed with the stdlib `zipfile` + `xml.etree`; odfpy was dropped in v1.0.1 because it is sdist-only and fails to build under Debian's patched setuptools.) **Each format parser must guard on its dep being non-None and raise/skip gracefully if not** - importing at module top-level would break `pip install -e .` minimal installs. When adding a new optional-format parser, follow the same pattern (top-of-file `try: import X; except: X = None`, then guard inside the `_parse_<ext>` method).

The `pyproject.toml` `dependencies` list installs every parser dep by default; the docs market boto3, PyPDF2, etc. as "optional" for users who want to skip them with `--no-deps` or a custom requirements file. Don't promote them to module-level imports.

## Demo (Cloudflare Workers AI)

`demo/` is a complete end-to-end example, not just sample docs:

- `demo/data/` - software-developer training corpus
- `demo/scripts/train_lora.sh|bat` - trains an adapter from the data
- `demo/scripts/deploy_to_r2.sh|bat` - uploads the adapter JSON to R2
- `demo/worker.js` + `demo/wrangler.toml` - Cloudflare Worker that loads the adapter and exposes `/chat`, `/chat/stream`, `/health`, `/docs`
- `demo/index.html` - browser UI for the Worker

The Worker binds `AI` (Workers AI) and `R2_BUCKET` (`doc2lora-adapters`). If you change the adapter JSON schema in `lora_trainer.save_adapter()`, the Worker's loading code in `demo/worker.js` must move in lockstep.

## Conventions specific to this repo

- **Black + isort with 88-char lines**; CI enforces both with `--check`. Don't reflow imports manually - run `isort` and `black` instead.
- **`mypy` is configured strict in `pyproject.toml`** (`disallow_untyped_defs = true`, `python_version = "3.8"`) but is not part of the CI workflow; if you turn it on locally, expect existing files to need annotations.
- **Logging over print**: modules use `logging.getLogger(__name__)`. CLI output uses `click.echo()`. Don't mix. **Library modules must NOT call `logging.basicConfig()`** - only `cli.py` configures root logging (a library-level `basicConfig` shadows the app's chosen format; fixed in v1.0.2). Keep internal/incidental logs at DEBUG so `scan` stays quiet.
- **Emoji in log messages is intentional** and matches across `core.py` / `lora_trainer.py` / `utils.py` (rocket for GPU, apple for MPS, laptop for CPU, etc.). Keep the convention if you add new device or stage logs; do not add emoji to anything user-facing in code comments or new identifiers.
- **`USAGE.md` and `README.md` overlap heavily** - when changing CLI flags or the public `convert*` signatures, update both, plus the relevant `examples/*.py` script.
