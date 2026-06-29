"""Tests for Cloudflare deploy helpers."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from doc2lora.deploy import (
    DEFAULT_CF_MODELS,
    deploy_adapter,
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


# --- deploy_adapter: wrangler backend (subprocess mocked) -------------------


def test_deploy_adapter_wrangler_success(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, model_type="mistral")
    with (
        patch("doc2lora.deploy.shutil.which", return_value="/usr/bin/wrangler"),
        patch("doc2lora.deploy.subprocess.run") as run,
    ):
        result = deploy_adapter(str(d), "myft", backend="wrangler")
    assert result == "myft"
    cmd = run.call_args[0][0]
    assert cmd[:4] == ["wrangler", "ai", "finetune", "create"]
    assert "myft" in cmd
    assert str(d) in cmd


def test_deploy_adapter_wrangler_missing_cli(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, model_type="mistral")
    with patch("doc2lora.deploy.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="wrangler"):
            deploy_adapter(str(d), "myft", backend="wrangler")


def test_deploy_adapter_unknown_backend(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, model_type="mistral")
    with pytest.raises(ValueError, match="Unknown backend"):
        deploy_adapter(str(d), "myft", backend="bogus")


def test_deploy_adapter_no_default_model(tmp_path):
    # an unrecognized model_type has no default cf_model and none was passed
    d = tmp_path / "adapter"
    _write_adapter(d, model_type="someunknownarch")
    with pytest.raises(ValueError, match="No default Cloudflare model"):
        deploy_adapter(str(d), "myft", backend="wrangler")


# --- deploy_adapter: REST backend (requests mocked) -------------------------


def test_deploy_adapter_rest_missing_creds(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, model_type="mistral")
    # inject a fake requests so the import inside _deploy_rest can't ImportError
    with patch.dict(sys.modules, {"requests": MagicMock()}):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="REST backend needs"):
                deploy_adapter(
                    str(d),
                    "myft",
                    backend="rest",
                    account_id=None,
                    api_token=None,
                )


def test_deploy_adapter_rest_success(tmp_path):
    d = tmp_path / "adapter"
    _write_adapter(d, model_type="mistral")

    mock_requests = MagicMock()
    resp = MagicMock()
    resp.json.return_value = {"result": {"id": "ft_123"}}
    mock_requests.post.return_value = resp

    with patch.dict(sys.modules, {"requests": mock_requests}):
        result = deploy_adapter(
            str(d),
            "myft",
            backend="rest",
            account_id="acct",
            api_token="tok",
        )
    assert result == "myft (ft_123)"
    # one create call + one upload per required file
    assert mock_requests.post.call_count == 3
