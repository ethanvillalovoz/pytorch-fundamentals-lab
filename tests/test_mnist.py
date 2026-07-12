import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lab.mnist as mnist_module
from pytorch_lab.mnist import MnistConfig, train_mnist


def test_mnist_smoke_run_uses_shared_metrics(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(7)
    images = torch.rand((24, 1, 28, 28), generator=generator)
    labels = torch.arange(24) % 10

    def fake_loaders(*args, **kwargs):
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=8), DataLoader(dataset, batch_size=6)

    monkeypatch.setattr(mnist_module, "build_mnist_loaders", fake_loaders)
    run = train_mnist(MnistConfig(epochs=1, batch_size=8, device="cpu"), download=False)
    assert len(run.history) == 1
    assert run.history[0].train.examples == 24
    assert run.history[0].test.examples == 24
    assert run.device == torch.device("cpu")


@pytest.mark.parametrize(
    "config",
    (
        MnistConfig(epochs=0),
        MnistConfig(learning_rate=0),
        MnistConfig(batch_size=0),
    ),
)
def test_invalid_mnist_config_is_rejected(config: MnistConfig) -> None:
    with pytest.raises(ValueError):
        train_mnist(config, download=False)
