"""MNIST data loading and training built on the shared metric loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lab.models import MnistCNN
from pytorch_lab.reproducibility import resolve_device, seed_everything
from pytorch_lab.training import ClassificationMetrics, evaluate, train_epoch


@dataclass(frozen=True)
class MnistConfig:
    seed: int = 42
    epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 128
    device: str | None = None

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class MnistEpoch:
    epoch: int
    train: ClassificationMetrics
    test: ClassificationMetrics


@dataclass
class MnistRun:
    model: MnistCNN
    device: torch.device
    history: list[MnistEpoch]


def build_mnist_loaders(
    data_dir: str | Path,
    *,
    batch_size: int,
    seed: int,
    download: bool,
) -> tuple[DataLoader, DataLoader]:
    """Load MNIST into one shared cache directory."""

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - exercised by users without the extra
        raise RuntimeError('MNIST support requires: pip install -e ".[vision]"') from exc

    transform = transforms.Compose(
        (transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)))
    )
    root = str(Path(data_dir))
    train_dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_mnist(
    config: MnistConfig | None = None,
    *,
    data_dir: str | Path = "data",
    download: bool = True,
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
) -> MnistRun:
    """Train the reference CNN, optionally limiting batches for smoke tests."""

    config = config or MnistConfig()
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_loader, test_loader = build_mnist_loaders(
        data_dir,
        batch_size=config.batch_size,
        seed=config.seed,
        download=download,
    )
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[MnistEpoch] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=max_train_batches,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            max_batches=max_test_batches,
        )
        history.append(MnistEpoch(epoch=epoch, train=train_metrics, test=test_metrics))
    return MnistRun(model=model, device=device, history=history)
