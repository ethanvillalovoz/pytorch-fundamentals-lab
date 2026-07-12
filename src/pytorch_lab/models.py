"""Small reference models used throughout the lab."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class IrisMLP(nn.Module):
    """A compact multilayer perceptron for the four Iris features."""

    def __init__(
        self,
        input_features: int = 4,
        hidden_features: Sequence[int] = (16, 8),
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if not hidden_features:
            raise ValueError("hidden_features must contain at least one layer")

        layers: list[nn.Module] = []
        previous = input_features
        for width in hidden_features:
            if width <= 0:
                raise ValueError("hidden layer widths must be positive")
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits."""

        return self.network(features)


class MnistCNN(nn.Module):
    """A small CNN that maps 28 by 28 grayscale images to ten logits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits; cross entropy applies softmax internally."""

        return self.classifier(self.features(images))
