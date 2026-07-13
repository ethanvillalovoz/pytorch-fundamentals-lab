"""Render a restrained reproducibility figure from the committed Iris run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "iris-reference" / "metrics.json"
OUTPUT = ROOT / "docs" / "media" / "iris-reference-evidence.png"

INK = "#202124"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#315bd6"
ERROR = "#c85b43"


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    losses = np.asarray(metrics["train_loss"], dtype=float)
    matrix = np.asarray(metrics["test"]["confusion_matrix"], dtype=int)
    names = metrics["dataset"]["class_names"]

    figure, (loss_axis, matrix_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5.6),
        gridspec_kw={"width_ratios": (1.45, 1)},
        facecolor="white",
    )
    epochs = np.arange(1, losses.size + 1)
    loss_axis.plot(epochs, losses, color=BLUE, linewidth=2.2)
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.set_title("Training loss", loc="left", fontsize=13, fontweight=700)
    loss_axis.grid(axis="y", color=GRID, linewidth=0.8)
    loss_axis.spines[["top", "right"]].set_visible(False)
    loss_axis.text(
        0.98,
        0.93,
        f"final loss  {losses[-1]:.4f}",
        transform=loss_axis.transAxes,
        ha="right",
        color=MUTED,
        fontsize=9,
    )

    image = matrix_axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    del image
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            matrix_axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=("white" if value > matrix.max() / 2 else (ERROR if value else MUTED)),
                fontsize=16,
                fontweight=700,
            )
    matrix_axis.set_xticks(np.arange(len(names)), names, rotation=24, ha="right")
    matrix_axis.set_yticks(np.arange(len(names)), names)
    matrix_axis.set_xlabel("Predicted class")
    matrix_axis.set_ylabel("True class")
    matrix_axis.set_title("Held-out confusion matrix", loc="left", fontsize=13, fontweight=700)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)

    figure.suptitle(
        "Iris MLP reference run",
        x=0.07,
        y=0.98,
        ha="left",
        color=INK,
        fontsize=17,
        fontweight=700,
    )
    train_examples = metrics["dataset"]["train_examples"]
    test_examples = metrics["dataset"]["test_examples"]
    figure.text(
        0.07,
        0.02,
        f"seed {metrics['config']['seed']}  ·  "
        f"{train_examples} train / {test_examples} test  ·  "
        f"{metrics['model']['parameters']} parameters  ·  "
        f"held-out accuracy {metrics['test']['accuracy']:.1%}",
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.18, wspace=0.28)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=160, facecolor="white")
    plt.close(figure)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
