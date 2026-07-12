"""Generate the committed, output-free lesson notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def cell_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id(source),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(source),
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


LESSONS: dict[str, list[dict[str, object]]] = {
    "01_tensors.ipynb": [
        markdown(
            "# Tensors\n\n"
            "Start with shape, dtype, and device. Those three properties explain most tensor "
            "errors before a model is involved. Install the lab first with `pip install -e .`."
        ),
        code("import torch\n\ntorch.manual_seed(42)"),
        code(
            "features = torch.tensor([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]])\n"
            "print(features)\n"
            'print(f"shape={tuple(features.shape)}, dtype={features.dtype}, '
            'device={features.device}")'
        ),
        markdown(
            "## Construct deliberately\n\n"
            "Use constructors that make intent visible. Model inputs are usually floating point; "
            "class labels passed to cross entropy are integer indices."
        ),
        code(
            "inputs = torch.zeros((3, 4), dtype=torch.float32)\n"
            "labels = torch.tensor([0, 1, 2], dtype=torch.long)\n"
            "random_batch = torch.rand((3, 4))\n"
            "inputs.shape, labels.dtype, random_batch.mean()"
        ),
        markdown(
            "## Move data and models together\n\n"
            "The package uses one device resolver so examples do not silently mix CPU and GPU "
            "tensors."
        ),
        code(
            "from pytorch_lab.reproducibility import resolve_device\n\n"
            "device = resolve_device()\n"
            "features_on_device = features.to(device)\n"
            "print(device, features_on_device.device)"
        ),
    ],
    "02_tensor_operations.ipynb": [
        markdown(
            "# Tensor operations\n\n"
            "Reshape, slice, broadcast, and combine tensors while checking shape at every step."
        ),
        code("import torch\n\nx = torch.arange(24).reshape(2, 3, 4)\nx"),
        markdown(
            "## `view` and `reshape`\n\n"
            "`view` requires contiguous storage. `reshape` may copy when it cannot return a view."
        ),
        code(
            "matrix = x.reshape(6, 4)\n"
            "transposed = matrix.T\n"
            "print(matrix.is_contiguous(), transposed.is_contiguous())\n"
            "print(transposed.reshape(2, 12).shape)"
        ),
        markdown("## Slicing and broadcasting"),
        code(
            "first_channel = x[:, :, 0]\n"
            "offsets = torch.tensor([0, 10, 20, 30])\n"
            "shifted = x + offsets\n"
            "first_channel, shifted[0]"
        ),
        markdown("## Concatenate versus stack"),
        code(
            "a = torch.ones((2, 3))\n"
            "b = torch.zeros((2, 3))\n"
            'print("cat", torch.cat((a, b), dim=0).shape)\n'
            'print("stack", torch.stack((a, b), dim=0).shape)'
        ),
    ],
    "03_autograd.ipynb": [
        markdown(
            "# Tensor math and autograd\n\n"
            "PyTorch records operations on tensors with `requires_grad=True`, then applies the "
            "chain "
            "rule when `backward()` is called."
        ),
        code(
            "import torch\n\n"
            "weight = torch.tensor(2.0, requires_grad=True)\n"
            "feature = torch.tensor(3.0)\n"
            "target = torch.tensor(10.0)\n"
            "prediction = weight * feature\n"
            "loss = (prediction - target) ** 2\n"
            "loss.backward()\n"
            'print(f"loss={loss.item():.1f}, dloss/dweight={weight.grad.item():.1f}")'
        ),
        markdown(
            "## A complete update\n\n"
            "Optimizers zero gradients, backpropagate one scalar loss, then update parameters."
        ),
        code(
            "model = torch.nn.Linear(1, 1)\n"
            "optimizer = torch.optim.SGD(model.parameters(), lr=0.05)\n"
            "criterion = torch.nn.MSELoss()\n"
            "x = torch.tensor([[1.0], [2.0], [3.0]])\n"
            "y = 2 * x + 1\n\n"
            "for _ in range(100):\n"
            "    optimizer.zero_grad(set_to_none=True)\n"
            "    batch_loss = criterion(model(x), y)\n"
            "    batch_loss.backward()\n"
            "    optimizer.step()\n\n"
            "with torch.inference_mode():\n"
            "    print(model(torch.tensor([[4.0]])).item())"
        ),
        markdown(
            "## Inference mode\n\n"
            "Use `torch.inference_mode()` for evaluation. It avoids building a graph and makes the "
            "intent explicit."
        ),
    ],
    "04_iris_mlp.ipynb": [
        markdown(
            "# Iris multilayer perceptron\n\n"
            "This notebook calls the same deterministic experiment that the CLI and test suite "
            "use. "
            "The scaler is fit on training data only and the split is stratified."
        ),
        code(
            "from pytorch_lab.iris import IrisConfig, train_iris\n\n"
            "run = train_iris(IrisConfig(seed=42, epochs=250))\n"
            'print(f"test accuracy: {run.test_accuracy:.1%}")\n'
            'print(f"test loss: {run.test_loss:.4f}")\n'
            "print(run.confusion)"
        ),
        markdown("## Inspect the learning curve"),
        code(
            "import matplotlib.pyplot as plt\n\n"
            "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
            'ax.plot(run.train_loss, color="#2563eb")\n'
            'ax.set(xlabel="epoch", ylabel="cross entropy", title="Iris MLP training loss")\n'
            'ax.spines[["top", "right"]].set_visible(False)\n'
            "plt.show()"
        ),
        markdown("## Inspect individual predictions"),
        code(
            "for label, prediction in zip(run.split.test_labels.tolist(), "
            "run.predictions.tolist(), strict=True):\n"
            "    print(\n"
            '        f"label={run.split.class_names[label]:10s} '
            'prediction={run.split.class_names[prediction]}"\n'
            "    )"
        ),
        markdown(
            "Run `torch-lab iris --output-dir artifacts/iris` to serialize metrics and predictions."
        ),
    ],
    "05_mnist_cnn.ipynb": [
        markdown(
            "# MNIST convolutional network\n\n"
            "The CNN returns logits, uses cross entropy directly, and aggregates loss and accuracy "
            "over examples rather than averaging batch summaries. MNIST is downloaded into one "
            "ignored cache directory."
        ),
        code(
            "from pytorch_lab.models import MnistCNN\n\n"
            "model = MnistCNN()\n"
            "print(model)\n"
            'print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")'
        ),
        markdown("## Run a bounded smoke experiment"),
        code(
            "from pytorch_lab.mnist import MnistConfig, train_mnist\n\n"
            "run = train_mnist(\n"
            "    MnistConfig(epochs=1, batch_size=128),\n"
            '    data_dir="../data",\n'
            "    max_train_batches=100,\n"
            "    max_test_batches=20,\n"
            ")\n"
            "last = run.history[-1]\n"
            'print(f"device: {run.device}")\n'
            'print(f"train: loss={last.train.loss:.4f}, accuracy={last.train.accuracy:.1%}")\n'
            'print(f"test:  loss={last.test.loss:.4f}, accuracy={last.test.accuracy:.1%}")'
        ),
        markdown(
            "Remove the batch limits for a full run, or use `torch-lab mnist --epochs 3`. The "
            "repository does not claim a benchmark result without a committed, reproducible run."
        ),
    ],
}


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    expected = set(LESSONS)
    for path in NOTEBOOKS.glob("*.ipynb"):
        if path.name not in expected:
            path.unlink()
    for filename, cells in LESSONS.items():
        destination = NOTEBOOKS / filename
        destination.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
