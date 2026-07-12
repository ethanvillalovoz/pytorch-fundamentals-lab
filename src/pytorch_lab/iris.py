"""Deterministic Iris experiment and artifact serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from pytorch_lab.models import IrisMLP
from pytorch_lab.reproducibility import seed_everything


@dataclass(frozen=True)
class IrisConfig:
    """Configuration for the deterministic CPU reference experiment."""

    seed: int = 42
    epochs: int = 250
    learning_rate: float = 0.01
    test_size: float = 0.2
    hidden_features: tuple[int, ...] = (16, 8)

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between zero and one")


@dataclass(frozen=True)
class IrisSplit:
    train_features: torch.Tensor
    test_features: torch.Tensor
    train_labels: torch.Tensor
    test_labels: torch.Tensor
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    class_names: tuple[str, ...]


@dataclass
class IrisRun:
    config: IrisConfig
    model: IrisMLP
    split: IrisSplit
    train_loss: list[float]
    test_loss: float
    test_accuracy: float
    predictions: torch.Tensor
    confusion: list[list[int]]


def load_iris_split(seed: int = 42, test_size: float = 0.2) -> IrisSplit:
    """Create a stratified split and fit scaling on training data only."""

    dataset = load_iris()
    features = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)
    return IrisSplit(
        train_features=torch.from_numpy(train_x),
        test_features=torch.from_numpy(test_x),
        train_labels=torch.from_numpy(train_y),
        test_labels=torch.from_numpy(test_y),
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        class_names=tuple(str(name) for name in dataset.target_names),
    )


def train_iris(config: IrisConfig | None = None) -> IrisRun:
    """Train and evaluate the reference MLP entirely on CPU."""

    config = config or IrisConfig()
    config.validate()
    seed_everything(config.seed)
    split = load_iris_split(config.seed, config.test_size)
    model = IrisMLP(hidden_features=config.hidden_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(split.train_features)
        loss = criterion(logits, split.train_labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    model.eval()
    with torch.inference_mode():
        test_logits = model(split.test_features)
        test_loss = float(criterion(test_logits, split.test_labels).item())
        predictions = test_logits.argmax(dim=1)
    accuracy = float((predictions == split.test_labels).float().mean().item())
    confusion = confusion_matrix(split.test_labels.numpy(), predictions.numpy()).tolist()
    return IrisRun(
        config=config,
        model=model,
        split=split,
        train_loss=history,
        test_loss=test_loss,
        test_accuracy=accuracy,
        predictions=predictions,
        confusion=confusion,
    )


def write_iris_artifacts(
    run: IrisRun,
    output_dir: str | Path,
    *,
    include_checkpoint: bool = False,
) -> tuple[Path, Path]:
    """Write deterministic metrics and per-example predictions."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    predictions_path = output / "predictions.csv"
    parameter_count = sum(parameter.numel() for parameter in run.model.parameters())
    payload = {
        "dataset": {
            "name": "Iris",
            "train_examples": int(run.split.train_labels.shape[0]),
            "test_examples": int(run.split.test_labels.shape[0]),
            "class_names": list(run.split.class_names),
        },
        "config": asdict(run.config),
        "model": {"name": "IrisMLP", "parameters": parameter_count},
        "test": {
            "loss": run.test_loss,
            "accuracy": run.test_accuracy,
            "confusion_matrix": run.confusion,
        },
        "train_loss": run.train_loss,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("example", "label", "prediction", "correct"))
        for index, (label, prediction) in enumerate(
            zip(run.split.test_labels.tolist(), run.predictions.tolist(), strict=True)
        ):
            writer.writerow((index, label, prediction, label == prediction))

    if include_checkpoint:
        torch.save(
            {
                "state_dict": run.model.state_dict(),
                "config": asdict(run.config),
                "scaler_mean": run.split.scaler_mean,
                "scaler_scale": run.split.scaler_scale,
                "class_names": run.split.class_names,
                "test_accuracy": run.test_accuracy,
            },
            output / "iris_mlp.pt",
        )
    return metrics_path, predictions_path
