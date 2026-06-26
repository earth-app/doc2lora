"""Tests for Cloudflare deploy helpers."""

import json

import pytest

from doc2lora.deploy import (
    DEFAULT_CF_MODELS,
    resolve_adapter_dir,
    validate_adapter,
)


def _write_adapter(directory, rank=8, model_type="mistral", size_bytes=10):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": rank, "model_type": model_type})
    )
    (directory / "adapter_model.safetensors").write_bytes(b"x" * size_bytes)


def test_validate_adapter_ok(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, rank=16)
    info = validate_adapter(str(d))
    assert info["rank"] == 16
    assert info["model_type"] == "mistral"


def test_validate_adapter_rank_too_high(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, rank=64)
    with pytest.raises(ValueError, match="rank"):
        validate_adapter(str(d))


def test_validate_adapter_missing_file(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 8, "model_type": "llama"}))
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        validate_adapter(str(d))


def test_validate_adapter_missing_model_type(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": 8}))
    (d / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="model_type"):
        validate_adapter(str(d))


def test_resolve_adapter_dir_from_json(tmp_path):
    d = tmp_path / "myadapter"
    _write_adapter(d)
    meta = tmp_path / "adapter.json"
    meta.write_text(json.dumps({"adapter_path": str(d)}))

    resolved = resolve_adapter_dir(str(meta))
    assert resolved == d


def test_resolve_adapter_dir_from_directory(tmp_path):
    d = tmp_path / "myadapter"
    _write_adapter(d)
    assert resolve_adapter_dir(str(d)) == d


def test_default_cf_models_use_mistralai():
    assert DEFAULT_CF_MODELS["mistral"] == "@cf/mistralai/mistral-7b-instruct-v0.2-lora"


def test_default_cf_models_include_qwq():
    assert DEFAULT_CF_MODELS["qwen"] == "@cf/qwen/qwq-32b"


def test_validate_adapter_accepts_qwen_model_type(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, rank=16, model_type="qwen")
    info = validate_adapter(str(d))
    assert info["model_type"] == "qwen"
