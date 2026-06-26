"""Tests for the LoRA trainer (model load is mocked out)."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from doc2lora.lora_trainer import CLOUDFLARE_MAX_RANK, LoRATrainer, get_device


def _make_trainer(**kwargs):
    """Build a trainer without actually loading a model."""
    with patch.object(LoRATrainer, "_load_model", lambda self: None):
        return LoRATrainer(device="cpu", **kwargs)


def test_get_device_returns_device():
    import torch

    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "mps", "cpu")


def test_cloudflare_model_type_detection():
    assert (
        _make_trainer(model_name="mistralai/Mistral-7B")._get_cloudflare_model_type()
        == "mistral"
    )
    assert (
        _make_trainer(model_name="google/gemma-7b-it")._get_cloudflare_model_type()
        == "gemma"
    )
    assert (
        _make_trainer(model_name="meta-llama/Llama-2-7b")._get_cloudflare_model_type()
        == "llama"
    )
    assert (
        _make_trainer(model_name="Qwen/QwQ-32B")._get_cloudflare_model_type() == "qwen"
    )


def test_find_target_modules_known_models():
    t = _make_trainer(model_name="mistralai/Mistral-7B-Instruct-v0.2")
    assert t._find_target_modules() == ["q_proj", "k_proj", "v_proj", "o_proj"]
    t2 = _make_trainer(model_name="meta-llama/Llama-2-7b")
    assert t2._find_target_modules() == ["q_proj", "v_proj"]
    t3 = _make_trainer(model_name="Qwen/QwQ-32B")
    assert t3._find_target_modules() == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_rank_warning_above_ceiling(caplog):
    with caplog.at_level(logging.WARNING):
        _make_trainer(lora_r=CLOUDFLARE_MAX_RANK + 8)
    assert "exceeds" in caplog.text


def test_rank_info_between_8_and_ceiling(caplog):
    with caplog.at_level(logging.INFO):
        _make_trainer(lora_r=16)
    assert "higher-rank" in caplog.text


def test_rank_8_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        _make_trainer(lora_r=8)
    assert "exceeds" not in caplog.text


def test_save_adapter_json_schema(tmp_path):
    t = _make_trainer(model_name="mistralai/Mistral-7B-Instruct-v0.2", lora_r=8)
    t.target_modules = ["q_proj", "v_proj"]
    t.peft_model = MagicMock()

    def fake_save(adapter_dir):
        d = Path(adapter_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16}))

    t.peft_model.save_pretrained.side_effect = fake_save

    out = tmp_path / "adapter.json"
    result = t.save_adapter(str(out))

    assert result == str(out)
    metadata = json.loads(out.read_text())
    assert metadata["model_type"] == "mistral"
    assert metadata["base_model"] == "mistralai/Mistral-7B-Instruct-v0.2"
    assert metadata["lora_config"]["r"] == 8
    assert metadata["cloudflare_compatible"] is True

    # model_type was injected into the adapter_config.json too
    cfg = json.loads((tmp_path / "adapter_adapter" / "adapter_config.json").read_text())
    assert cfg["model_type"] == "mistral"
