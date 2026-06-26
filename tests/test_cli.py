"""Tests for the doc2lora CLI."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from doc2lora import __version__
from doc2lora.cli import cli


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_formats_lists_new_extensions():
    result = CliRunner().invoke(cli, ["formats"])
    assert result.exit_code == 0
    for token in [".pptx", ".epub", ".ipynb", ".7z", "source code"]:
        assert token in result.output


def test_scan_reports_estimate(tmp_path):
    (tmp_path / "a.md").write_text("# title\n\nbody")
    (tmp_path / "b.py").write_text("print('hi')")

    result = CliRunner().invoke(cli, ["scan", str(tmp_path), "--device", "cpu"])
    assert result.exit_code == 0
    assert "Found 2 supported documents" in result.output
    assert "Estimated training time" in result.output


@patch("doc2lora.core.LoRATrainer")
def test_convert_invokes_trainer(mock_trainer_class, tmp_path):
    mock_trainer = MagicMock()
    mock_trainer.save_adapter.return_value = str(tmp_path / "out.json")
    mock_trainer_class.return_value = mock_trainer

    (tmp_path / "doc.md").write_text("# hi\n\ncontent")
    out = tmp_path / "out.json"

    result = CliRunner().invoke(
        cli,
        ["convert", str(tmp_path), "-o", str(out), "--device", "cpu", "--epochs", "1"],
    )
    assert result.exit_code == 0, result.output
    mock_trainer_class.assert_called_once()
    mock_trainer.train.assert_called_once()


@patch("doc2lora.deploy.deploy_adapter")
def test_deploy_command(mock_deploy, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    mock_deploy.return_value = "my-finetune"

    result = CliRunner().invoke(
        cli,
        ["deploy", str(adapter_dir), "my-finetune", "--cf-model", "@cf/x/y-lora"],
    )
    assert result.exit_code == 0, result.output
    mock_deploy.assert_called_once()
    assert "Deployed" in result.output
