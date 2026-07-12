import json
from types import SimpleNamespace

import pytest

import pytorch_lab.cli as cli_module
from pytorch_lab.cli import build_parser, main
from pytorch_lab.training import ClassificationMetrics


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_iris_cli_writes_artifacts(tmp_path, capsys) -> None:
    assert main(["iris", "--epochs", "40", "--output-dir", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert output["metrics"].endswith("metrics.json")


def test_mnist_cli_serializes_epoch_metrics(monkeypatch, capsys) -> None:
    metrics = ClassificationMetrics(loss=0.5, accuracy=0.75, examples=8)
    fake_run = SimpleNamespace(history=[SimpleNamespace(epoch=1, train=metrics, test=metrics)])
    monkeypatch.setattr(cli_module, "train_mnist", lambda *args, **kwargs: fake_run)

    assert main(["mnist", "--epochs", "1", "--no-download"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == [
        {
            "epoch": 1,
            "train": {"loss": 0.5, "accuracy": 0.75, "examples": 8},
            "test": {"loss": 0.5, "accuracy": 0.75, "examples": 8},
        }
    ]
