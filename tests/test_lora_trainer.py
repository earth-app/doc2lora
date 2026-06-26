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


def test_select_optim_cpu():
    # fused AdamW is CUDA-only; CPU stays on plain AdamW
    assert _make_trainer()._select_optim() == "adamw_torch"


def _build_args(**overrides):
    t = _make_trainer()
    kwargs = dict(
        output_dir="./out",
        batch_size=2,
        num_epochs=1,
        max_steps=None,
        learning_rate=5e-4,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        optim="adamw_torch",
        group_by_length=False,
        dataloader_num_workers=None,
        torch_compile=False,
    )
    kwargs.update(overrides)
    return t._build_training_arguments(**kwargs)


def test_build_training_arguments_cpu_defaults():
    args = _build_args()
    # cpu -> fp32 (no bf16/fp16), no pinned host memory, plain AdamW
    assert args.bf16 is False
    assert args.fp16 is False
    assert args.dataloader_pin_memory is False
    assert args.optim == "adamw_torch"
    assert args.gradient_checkpointing is True
    assert args.num_train_epochs == 1


def test_build_training_arguments_group_by_length_and_workers():
    args = _build_args(group_by_length=True, dataloader_num_workers=2)
    # robust across transformers 4.x (bool) and 5.x (sampler strategy)
    grouped = getattr(args, "group_by_length", False) or (
        getattr(args, "train_sampling_strategy", None) == "group_by_length"
    )
    assert grouped
    assert args.dataloader_num_workers == 2


def test_chunk_texts_windows_long_documents():
    t = _make_trainer()  # _load_model is mocked, so wire a fake tokenizer
    t.max_length = 10
    t.chunk_overlap = 0
    # fake tokenizer: one token per character
    t.tokenizer = lambda text, **kwargs: {"input_ids": list(range(len(text)))}

    # a document at/under max_length stays a single example
    ids, attn = t._chunk_texts(["abcdefgh"])  # 8 tokens
    assert [len(c) for c in ids] == [8]
    assert attn == [[1] * 8]

    # a 28-token document windows into 3 chunks (10, 10, 8); none exceed max_length
    ids, _ = t._chunk_texts(["x" * 28])
    assert [len(c) for c in ids] == [10, 10, 8]
    assert all(len(c) <= t.max_length for c in ids)

    # overlap shortens the step, producing more (overlapping) chunks
    t.chunk_overlap = 5
    ids_overlap, _ = t._chunk_texts(["x" * 28])
    assert len(ids_overlap) > 3


def test_chunk_texts_disabled_is_truncation_default_changed():
    # chunking is on by default
    assert _make_trainer().chunk_long_documents is True


def test_should_group_by_length_auto_heuristic():
    t = _make_trainer()
    # explicit override always wins
    assert t._should_group_by_length(True, 1) is True
    assert t._should_group_by_length(False, 8) is False
    # auto (None): only when batch_size >= 2 (no padding to cut at batch 1)
    assert t._should_group_by_length(None, 1) is False
    assert t._should_group_by_length(None, 2) is True
    assert t._should_group_by_length(None, 8) is True


def test_should_compile_auto_heuristic():
    import torch

    from doc2lora.lora_trainer import TORCH_COMPILE_AUTO_MIN_CHARS

    t = _make_trainer()  # device is cpu
    small = [{"content": "x" * 100}]
    big = [{"content": "x" * (TORCH_COMPILE_AUTO_MIN_CHARS + 1)}]

    # an explicit True/False always wins over the heuristic
    assert t._should_compile(True, small) is True
    assert t._should_compile(False, big) is False

    # auto (None) never compiles on CPU, even for a big corpus
    assert t._should_compile(None, big) is False

    # auto (None) compiles on CUDA only when the corpus is large enough
    t.device = torch.device("cuda")
    assert t._should_compile(None, big) is True
    assert t._should_compile(None, small) is False
