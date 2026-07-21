"""Render a restrained reproducibility figure from the committed Iris run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "iris-reference" / "metrics.json"
PREDICTIONS_PATH = ROOT / "artifacts" / "iris-reference" / "predictions.csv"
OUTPUT_STEM = ROOT / "docs" / "media" / "iris-reference-evidence"

INK = "#202124"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BLUE = "#315bd6"
ERROR = "#c85b43"
GREEN = "#2f7d4a"


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    with PREDICTIONS_PATH.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    losses = np.asarray(metrics["train_loss"], dtype=float)
    matrix = np.asarray(metrics["test"]["confusion_matrix"], dtype=int)
    names = metrics["dataset"]["class_names"]
    errors = [row for row in predictions if row["correct"] == "False"]
    if len(errors) != 2:
        raise RuntimeError(f"expected two committed Iris errors, found {len(errors)}")

    figure, (loss_axis, matrix_axis, audit_axis) = plt.subplots(
        1,
        3,
        figsize=(14, 5.6),
        gridspec_kw={"width_ratios": (1.35, 0.95, 0.85)},
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

    audit_axis.axis("off")
    audit_axis.set_title("Error audit", loc="left", fontsize=13, fontweight=700, pad=8)
    audit_axis.text(
        0,
        0.88,
        "28 / 30 correct",
        color=GREEN,
        fontsize=15,
        fontweight=700,
        transform=audit_axis.transAxes,
    )
    audit_axis.text(
        0,
        0.79,
        "Errors cross only the\nversicolor–virginica boundary.",
        color=MUTED,
        fontsize=9.5,
        linespacing=1.4,
        transform=audit_axis.transAxes,
    )
    for index, row in enumerate(errors):
        top = 0.59 - index * 0.24
        label = names[int(row["label"])]
        prediction = names[int(row["prediction"])]
        audit_axis.text(
            0,
            top,
            f"sample {row['example']}",
            color=MUTED,
            fontsize=9,
            fontweight=700,
            transform=audit_axis.transAxes,
        )
        audit_axis.text(
            0,
            top - 0.09,
            label,
            color=INK,
            fontsize=11,
            fontweight=700,
            transform=audit_axis.transAxes,
        )
        audit_axis.text(
            0.48,
            top - 0.09,
            f"→  {prediction}",
            color=ERROR,
            fontsize=11,
            fontweight=700,
            transform=audit_axis.transAxes,
        )
        audit_axis.plot(
            [0, 1],
            [top - 0.15, top - 0.15],
            color=GRID,
            linewidth=1,
            transform=audit_axis.transAxes,
        )
    audit_axis.text(
        0,
        0.06,
        "One seed · no uncertainty estimate",
        color=MUTED,
        fontsize=9,
        transform=audit_axis.transAxes,
    )

    figure.suptitle(
        "Iris MLP reference run · learning trace and held-out errors",
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
    figure.subplots_adjust(left=0.06, right=0.985, top=0.87, bottom=0.18, wspace=0.32)
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=200, facecolor="white")
    plt.close(figure)
    print(f"Wrote {OUTPUT_STEM.relative_to(ROOT)}.[png|svg|pdf]")


if __name__ == "__main__":
    main()
