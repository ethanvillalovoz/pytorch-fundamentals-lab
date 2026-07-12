"""Shared supervised-learning loops with sample-weighted metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationMetrics:
    """Loss and accuracy aggregated over individual examples."""

    loss: float
    accuracy: float
    examples: int


def _run_batches(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> ClassificationMetrics:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (features, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            features = features.to(device)
            labels = labels.to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    if total_examples == 0:
        raise ValueError("loader did not yield any examples")
    return ClassificationMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
        examples=total_examples,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> ClassificationMetrics:
    """Train for one pass and return sample-weighted metrics."""

    return _run_batches(
        model,
        loader,
        criterion,
        device,
        optimizer=optimizer,
        max_batches=max_batches,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> ClassificationMetrics:
    """Evaluate without gradients and return sample-weighted metrics."""

    return _run_batches(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
        max_batches=max_batches,
    )
