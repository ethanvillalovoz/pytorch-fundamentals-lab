import json

import numpy as np
import pytest
import torch

from pytorch_lab.iris import IrisConfig, load_iris_split, train_iris, write_iris_artifacts


def test_split_is_stratified_and_train_scaled() -> None:
    split = load_iris_split(seed=42)
    assert split.train_features.shape == (120, 4)
    assert split.test_features.shape == (30, 4)
    assert torch.bincount(split.test_labels).tolist() == [10, 10, 10]
    np.testing.assert_allclose(split.train_features.mean(dim=0).numpy(), 0, atol=1e-6)


def test_reference_run_learns_and_is_deterministic() -> None:
    config = IrisConfig(epochs=150)
    first = train_iris(config)
    second = train_iris(config)
    assert first.test_accuracy >= 0.9
    assert first.train_loss[-1] < first.train_loss[0] * 0.2
    assert first.train_loss == second.train_loss
    assert first.confusion == second.confusion
    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_artifacts_capture_metrics_and_predictions(tmp_path) -> None:
    run = train_iris(IrisConfig(epochs=80))
    metrics_path, predictions_path = write_iris_artifacts(
        run,
        tmp_path,
        include_checkpoint=True,
    )
    metrics = json.loads(metrics_path.read_text())
    rows = predictions_path.read_text().splitlines()
    checkpoint = torch.load(tmp_path / "iris_mlp.pt", weights_only=True)

    assert metrics["model"]["parameters"] > 0
    assert metrics["test"]["accuracy"] == run.test_accuracy
    assert len(rows) == 31
    assert checkpoint["test_accuracy"] == run.test_accuracy


@pytest.mark.parametrize(
    "config",
    (
        IrisConfig(epochs=0),
        IrisConfig(learning_rate=0),
        IrisConfig(test_size=1),
    ),
)
def test_invalid_iris_config_is_rejected(config: IrisConfig) -> None:
    with pytest.raises(ValueError):
        train_iris(config)
